"""
Mind class for the NeuralChild project.

This module contains the implementation of the Mind class which serves as the central hub
for integrating all neural components of the simulated child's mind and managing
interactions with the Mother component.

The Mind class implements the concept of an LMM (Large Mind Model), which integrates
various specialized neural networks to create a self-aware, emotive, and learning
artificial consciousness capable of developing through interactions.
"""

import os
import sys
import time
import json
import uuid
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
from collections import deque

# Add parent directory to path to import from sibling packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import neural components
from neural_child.cognition.cognitive_component import CognitiveComponent
from neural_child.emotion.emotional_component import EmotionalComponent
from neural_child.language.language_component import LanguageComponent
from neural_child.memory.memory_component import MemoryComponent
from neural_child.social.social_component import SocialComponent
from neural_child.development.development_component import DevelopmentComponent

# Import Mother component
from mother.mother import Mother

# Import base classes and utility functions
from neural_child.mind.base import NeuralComponent, MindState, InteractionState
from utils.config import NeuralChildConfig, DEFAULT_NEURAL_CHILD_CONFIG, MotherPersonalityConfig, DEFAULT_MOTHER_PERSONALITY, LLMConfig, DEFAULT_LLM_CONFIG


# Custom JSON encoder to handle numpy data types
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class Mind:
    """
    The central hub of the Neural Child system, implementing an LMM (Large Mind Model).
    
    The Mind class integrates all neural components and orchestrates their interactions,
    creating a unified cognitive system capable of learning, developing, and interacting
    with the Mother component. Unlike an LLM which only processes text, the LMM processes
    and generates multimodal experiences, emotions, and thoughts through an integrated 
    system of specialized neural networks.
    """
    
    def __init__(
        self,
        config: NeuralChildConfig = DEFAULT_NEURAL_CHILD_CONFIG,
        mother_personality: MotherPersonalityConfig = DEFAULT_MOTHER_PERSONALITY,
        llm_config: LLMConfig = DEFAULT_LLM_CONFIG,
        device: str = "cpu",
        base_path: Path = Path("data/neural_child"),
        load_existing: bool = False
    ):
        """
        Initialize the Mind with all neural components.
        
        Args:
            config: Configuration for the Neural Child
            mother_personality: Personality configuration for the Mother component
            llm_config: Configuration for the LLM (used only by Mother)
            device: Device to run neural components on ("cpu" or "cuda")
            base_path: Base path for saving/loading the Mind state
            load_existing: Whether to load existing models if available
        """
        self.config = config
        self.device = device
        self.base_path = base_path
        self.models_path = base_path / "models"
        self.interactions_path = base_path / "interactions"
        self.development_path = base_path / "development"
        self.lmm_path = base_path / "lmm"
        
        # Create directories if they don't exist
        for path in [self.models_path, self.interactions_path, self.development_path, self.lmm_path]:
            path.mkdir(parents=True, exist_ok=True)
            
        # Initialize mind state
        self.mind_state = MindState(age_months=config.initial_age_months)
        
        # Set up development component first as it guides other components
        self.development_component = DevelopmentComponent(
            initial_age_months=config.initial_age_months,
            development_speed=config.development_speed
        )
        
        # Initialize neural components
        self.cognitive_component = CognitiveComponent(
            learning_rate=config.learning_rate,
            device=device,
            name="cognitive_component"
        )
        
        self.emotional_component = EmotionalComponent(name="emotional_component")
        
        self.language_component = LanguageComponent(name="language_component")
        
        self.memory_component = MemoryComponent(
            memory_capacity=config.memory_capacity,
            name="memory_component"
        )
        
        self.social_component = SocialComponent(
            learning_rate=config.learning_rate,
            device=device,
            name="social_component"
        )
        
        # Patch social component save method to handle numpy data types
        original_social_save = self.social_component.save
        def patched_social_save(directory: Path):
            try:
                # Try original method first
                original_social_save(directory)
            except TypeError as e:
                if "not JSON serializable" in str(e):
                    print("Applying patch for social component save...")
                    
                    # Save model state
                    if hasattr(self.social_component, 'state_dict'):
                        torch.save(self.social_component.state_dict(), directory / f"{self.social_component.name}_state.pth")
                    
                    # Save additional state safely
                    if hasattr(self.social_component, 'relationship_models'):
                        model_path = directory / f"{self.social_component.name}_relationship_models.pth"
                        torch.save(self.social_component.relationship_models, model_path)
                    
                    # Save interaction history safely with numpy encoder
                    if hasattr(self.social_component, 'interaction_history'):
                        try:
                            history_path = directory / f"{self.social_component.name}_interaction_history.json"
                            with open(history_path, "w") as f:
                                json.dump(self.social_component.interaction_history, f, indent=2, cls=NumpyEncoder)
                        except Exception as inner_e:
                            print(f"Could not save interaction history: {inner_e}")
                            # Create empty history if cannot save existing one
                            with open(history_path, "w") as f:
                                json.dump([], f, indent=2)
                    print("Social component patched save completed")
                else:
                    # If it's a different error, raise it
                    raise e
        
        # Replace the method
        self.social_component.save = patched_social_save
        
        # Store all neural components in a dictionary for easier access
        self.components = {
            "cognitive": self.cognitive_component,
            "emotional": self.emotional_component,
            "language": self.language_component,
            "memory": self.memory_component,
            "social": self.social_component,
            "development": self.development_component
        }
        
        # Initialize the Mother component
        self.mother = Mother(
            personality=mother_personality,
            llm_config=llm_config,
            name="Mother"
        )
        
        # Training parameters
        self.learning_rate = config.learning_rate
        self.training_batch_size = 16
        self.training_buffer = []
        self.training_interval = 10  # Train after every 10 interactions
        self.interaction_count = 0
        
        # Last save timestamp
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes
        
        # LMM meta-learning parameters
        self.meta_learning_rate = 0.001
        self.meta_optimizer = torch.optim.Adam(
            self._get_all_parameters(), 
            lr=self.meta_learning_rate
        )
        
        # Interaction history
        self.interaction_history = deque(maxlen=100)
        
        # Load existing models if requested
        if load_existing:
            self.load()
        
        # Initial update of mind state
        self._update_mind_state()
    
    def _get_all_parameters(self) -> List[torch.nn.Parameter]:
        """Get all trainable parameters from all components."""
        parameters = []
        for component_name, component in self.components.items():
            if isinstance(component, NeuralComponent):
                parameters.extend(list(component.parameters()))
        return parameters
        
    def _update_mind_state(self) -> None:
        """Update the mind state with the current state of all components."""
        # Update development stage and age
        dev_state = self.development_component.update({})
        self.mind_state.age_months = dev_state.get("age_months", self.mind_state.age_months)
        self.mind_state.developmental_stage = dev_state.get("developmental_stage", self.mind_state.developmental_stage)
        
        # Update component activations
        for component_name, component in self.components.items():
            if isinstance(component, NeuralComponent):
                self.mind_state.component_activations[component_name] = component.activation_level
        
        # Update emotional state
        emotional_state = self.emotional_component.get_emotional_state()
        for emotion, value in emotional_state.items():
            self.mind_state.update_emotional_state(emotion, value)
        
        # Update language capabilities
        language_metrics = self.language_component.get_language_development_metrics()
        self.mind_state.language_comprehension = language_metrics.get("receptive_language_development", 0.0)
        self.mind_state.language_production = language_metrics.get("expressive_language_development", 0.0)
        self.mind_state.update_vocabulary_size(self.language_component.get_vocabulary_size())
        
        # Update working memory from memory component
        memory_counts = self.memory_component.get_memory_counts()
        memory_working = getattr(self.memory_component, "working_memory", [])
        for item in memory_working:
            self.mind_state.add_to_working_memory(item)
        
        # Update developmental metrics
        # Cognitive metrics
        cognitive_metrics = self.cognitive_component.get_developmental_metrics()
        for metric, value in cognitive_metrics.items():
            self.mind_state.update_developmental_metrics("cognitive", metric, value)
        
        # Emotional metrics
        emotional_metrics = self.emotional_component.get_emotional_development_metrics()
        for metric, value in emotional_metrics.items():
            self.mind_state.update_developmental_metrics("emotional", metric, value)
        
        # Language metrics
        for metric, value in language_metrics.items():
            if metric != "vocabulary_size":  # Already handled above
                self.mind_state.update_developmental_metrics("language", metric, value)
        
        # Social metrics
        social_metrics = self.social_component.get_social_development_metrics()
        for metric, value in social_metrics.items():
            self.mind_state.update_developmental_metrics("social", metric, value)
        
        # Memory metrics
        memory_metrics = self.memory_component.get_memory_development_metrics()
        for metric, value in memory_metrics.items():
            self.mind_state.update_developmental_metrics("cognitive", f"memory_{metric}", value)
    
    def interact_with_mother(self, context: Dict[str, Any] = None) -> InteractionState:
        """
        Process an interaction between the Neural Child and Mother.
        
        Args:
            context: Optional context for the interaction
        
        Returns:
            InteractionState object representing the interaction
        """
        # Update mind state before interaction
        self._update_mind_state()
        
        # Generate child utterance based on current state
        child_utterance = self._generate_child_utterance(context)
        
        try:
            # Get response from Mother
            interaction_state = self.mother.respond_to_child(
                child_state=self.mind_state,
                child_utterance=child_utterance
            )
            
            # Process mother's response through all components
            self._process_mother_response(interaction_state)
            
            # Add to interaction history
            self.interaction_history.append(interaction_state)
        except Exception as e:
            print(f"Error during mother interaction: {str(e)}")
            print("Creating a default interaction state instead")
            
            # Create a default interaction state
            interaction_state = InteractionState(
                interaction_id=str(uuid.uuid4()),
                timestamp=time.time(),
                age_months=self.mind_state.age_months,
                developmental_stage=self.mind_state.developmental_stage
            )
            
            # Add default mother response
            interaction_state.mother_state = {
                "verbal_response": "Hello, little one.",
                "emotional_state": {
                    "joy": 0.7,
                    "trust": 0.6,
                    "anticipation": 0.3
                },
                "teaching_elements": [
                    {
                        "type": "vocabulary",
                        "content": "hello",
                        "complexity": 0.1
                    }
                ]
            }
            
            # Add child state to interaction
            interaction_state.child_state = {
                "verbal_response": child_utterance,
                "emotional_state": self.mind_state.emotional_state,
                "attention_focus": self.mind_state.attention_focus,
                "needs": self.mind_state.needs
            }
            
            # Process the default response
            self._process_mother_response(interaction_state)
            
            # Add to interaction history
            self.interaction_history.append(interaction_state)
        
        # Increment interaction count and train if needed
        self.interaction_count += 1
        if self.interaction_count % self.training_interval == 0:
            self._train_components()
        
        # Save if enough time has passed
        current_time = time.time()
        if current_time - self.last_save_time > self.save_interval:
            self.save()
            self.last_save_time = current_time
        
        return interaction_state
    
    def _generate_child_utterance(self, context: Dict[str, Any] = None) -> str:
        """
        Generate an utterance from the child based on its current state.
        The child generates utterances through its language component, without
        using an external LLM.
        
        Args:
            context: Optional context for utterance generation
        
        Returns:
            String representing the child's utterance
        """
        if context is None:
            context = {}
        
        # Prepare input for language component
        inputs = {
            "developmental_stage": self.mind_state.developmental_stage,
            "age_months": self.mind_state.age_months,
            "emotional_state": self.mind_state.emotional_state
        }
        
        # Add relevant memories if available
        if hasattr(self, "memory_component"):
            query = {
                "content_type": "language",
                "recency": 0.7,
                "relevance": 0.3
            }
            relevant_memories = self.memory_component._retrieve_memories(query)
            inputs["relevant_memories"] = relevant_memories
        
        # Get output from language component
        language_output = self.language_component.process(inputs)
        
        # Extract and return child utterance
        child_utterance = language_output.get("child_utterance", "")
        return child_utterance
    
    def _process_mother_response(self, interaction_state: InteractionState) -> None:
        """
        Process the mother's response through all neural components.
        
        Args:
            interaction_state: InteractionState object representing the interaction
        """
        # Extract mother's verbal response and teaching elements
        mother_utterance = interaction_state.mother_state.get("verbal_response", "")
        mother_emotional_state = interaction_state.mother_state.get("emotional_state", {})
        teaching_elements = interaction_state.mother_state.get("teaching_elements", [])
        
        # Prepare input for components
        base_input = {
            "mother_utterance": mother_utterance,
            "mother_emotional_state": mother_emotional_state,
            "teaching_elements": teaching_elements,
            "developmental_stage": self.mind_state.developmental_stage,
            "age_months": self.mind_state.age_months,
            "interaction_id": interaction_state.interaction_id
        }
        
        # Process through cognitive component
        cognitive_input = base_input.copy()
        cognitive_input["emotional_state"] = self.mind_state.emotional_state
        cognitive_output = self.cognitive_component.process_input(cognitive_input)
        
        # Update cognitive activation and confidence
        self.cognitive_component.update_activation(cognitive_output.get("attention_level", 0.5))
        self.cognitive_component.update_confidence(cognitive_output.get("understanding_level", 0.5))
        
        # Process through emotional component
        emotional_input = base_input.copy()
        emotional_input["cognitive_output"] = cognitive_output
        emotional_output = self.emotional_component.process(emotional_input)
        
        # Update emotional state in mind state
        for emotion, value in emotional_output.get("emotional_state", {}).items():
            self.mind_state.update_emotional_state(emotion, value)
        
        # Process through language component
        language_input = base_input.copy()
        language_input["emotional_state"] = self.mind_state.emotional_state
        language_input["cognitive_output"] = cognitive_output
        language_output = self.language_component.process(language_input)
        
        # Process through memory component
        memory_input = {
            "experience": {
                "type": "interaction",
                "mother_utterance": mother_utterance,
                "teaching_elements": teaching_elements,
                "cognitive_output": cognitive_output,
                "language_output": language_output,
                "timestamp": time.time(),
                "age_months": self.mind_state.age_months,
                "developmental_stage": self.mind_state.developmental_stage
            },
            "emotional_state": self.mind_state.emotional_state
        }
        memory_output = self.memory_component.process(memory_input)
        
        # Process through social component
        social_input = {
            "agent": "mother",
            "content": mother_utterance,
            "emotional_tone": mother_emotional_state,
            "developmental_stage": self.mind_state.developmental_stage,
            "age_months": self.mind_state.age_months,
            "child_emotional_state": self.mind_state.emotional_state
        }
        social_output = self.social_component.process_interaction(social_input)
        
        # Add experience to training buffer for later training
        self.training_buffer.append({
            "input": base_input,
            "cognitive": {
                "input": cognitive_input,
                "output": cognitive_output
            },
            "emotional": {
                "input": emotional_input,
                "output": emotional_output
            },
            "language": {
                "input": language_input,
                "output": language_output
            },
            "memory": {
                "input": memory_input,
                "output": memory_output
            },
            "social": {
                "input": social_input,
                "output": social_output
            }
        })
        
        # Limit training buffer size
        if len(self.training_buffer) > 100:
            self.training_buffer = self.training_buffer[-100:]
    
    def _train_components(self) -> None:
        """Train all components using collected experiences from interactions."""
        if not self.training_buffer:
            return
        
        print(f"Training neural components after {self.interaction_count} interactions...")
        
        # Sample training data from buffer (or use all if less than batch size)
        if len(self.training_buffer) < self.training_batch_size:
            training_data = self.training_buffer
        else:
            training_data = np.random.choice(
                self.training_buffer, 
                self.training_batch_size, 
                replace=False
            ).tolist()
        
        # Train each component
        losses = {
            "cognitive": 0.0,
            "emotional": 0.0,
            "language": 0.0,
            "memory": 0.0,
            "social": 0.0
        }
        
        # Train cognitive component
        for data in training_data:
            # Cognitive training
            if isinstance(self.cognitive_component, NeuralComponent):
                cognitive_input = torch.tensor(
                    self._prepare_input_tensor(data["cognitive"]["input"]), 
                    dtype=torch.float32,
                    device=self.device
                )
                
                cognitive_target = torch.tensor(
                    self._prepare_output_tensor(data["cognitive"]["output"]),
                    dtype=torch.float32,
                    device=self.device
                )
                
                loss = self.cognitive_component.train_component(
                    cognitive_input, 
                    cognitive_target,
                    self.learning_rate
                )
                losses["cognitive"] += loss
                
            # Emotional training
            if isinstance(self.emotional_component, NeuralComponent):
                emotional_input = torch.tensor(
                    self._prepare_input_tensor(data["emotional"]["input"]),
                    dtype=torch.float32,
                    device=self.device
                )
                
                emotional_target = torch.tensor(
                    self._prepare_output_tensor(data["emotional"]["output"]),
                    dtype=torch.float32,
                    device=self.device
                )
                
                loss = self.emotional_component.train_component(
                    emotional_input,
                    emotional_target,
                    self.learning_rate
                )
                losses["emotional"] += loss
            
            # Language training
            if isinstance(self.language_component, NeuralComponent):
                language_input = torch.tensor(
                    self._prepare_input_tensor(data["language"]["input"]),
                    dtype=torch.float32,
                    device=self.device
                )
                
                language_target = torch.tensor(
                    self._prepare_output_tensor(data["language"]["output"]),
                    dtype=torch.float32,
                    device=self.device
                )
                
                loss = self.language_component.train_component(
                    language_input,
                    language_target,
                    self.learning_rate
                )
                losses["language"] += loss
            
            # Memory training
            if isinstance(self.memory_component, NeuralComponent):
                memory_input = torch.tensor(
                    self._prepare_input_tensor(data["memory"]["input"]),
                    dtype=torch.float32,
                    device=self.device
                )
                
                memory_target = torch.tensor(
                    self._prepare_output_tensor(data["memory"]["output"]),
                    dtype=torch.float32,
                    device=self.device
                )
                
                loss = self.memory_component.train_component(
                    memory_input,
                    memory_target,
                    self.learning_rate
                )
                losses["memory"] += loss
            
            # Social training
            if isinstance(self.social_component, NeuralComponent):
                social_input = torch.tensor(
                    self._prepare_input_tensor(data["social"]["input"]),
                    dtype=torch.float32,
                    device=self.device
                )
                
                social_target = torch.tensor(
                    self._prepare_output_tensor(data["social"]["output"]),
                    dtype=torch.float32,
                    device=self.device
                )
                
                loss = self.social_component.train_component(
                    social_input,
                    social_target,
                    self.learning_rate
                )
                losses["social"] += loss
            
        # Average losses
        for component in losses:
            if len(training_data) > 0:
                losses[component] /= len(training_data)
        
        print(f"Training losses: {losses}")
        
        # Meta-learning step to optimize integration between components
        self._meta_learning_step()
        
        # Update training progress for all components
        for component_name, component in self.components.items():
            if isinstance(component, NeuralComponent):
                component.update_training_progress(1.0)
                component.increment_experience()
    
    def _meta_learning_step(self) -> None:
        """
        Meta-learning step to optimize the connections between components.
        This represents the LMM (Large Mind Model) learning process that integrates
        all specialized neural networks.
        """
        # Skip if not enough interactions
        if self.interaction_count < 10:
            return
        
        # Zero gradients
        self.meta_optimizer.zero_grad()
        
        # Create LMM loss based on coherence between components
        # This promotes alignment between cognitive, emotional, language, memory
        # and social representations
        lmm_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Check if we have at least 3 interactions to calculate coherence
        if len(self.interaction_history) >= 3:
            # Get the last three interactions
            recent_interactions = list(self.interaction_history)[-3:]
            
            # Calculate coherence between component activations
            for i in range(len(recent_interactions) - 1):
                # This would involve extracting representations from each component
                # for the given interactions and calculating coherence/consistency
                # across components
                
                # As a simplified placeholder, we just use a constant loss
                # In a real implementation, this would be calculated based on
                # the alignment between component activations and outputs
                lmm_loss = lmm_loss + torch.tensor(0.1, device=self.device, requires_grad=True)
        
        # Backward pass
        lmm_loss.backward()
        
        # Update parameters
        self.meta_optimizer.step()
    
    def _prepare_input_tensor(self, input_data: Dict[str, Any]) -> List[float]:
        """
        Convert input data dictionary to a tensor.
        This is a simplified implementation - in a real system, this would
        handle complex input structures properly.
        
        Args:
            input_data: Dictionary of input data
            
        Returns:
            List of floats representing tensor data
        """
        # Simplified implementation - in reality, you'd need a more sophisticated
        # approach to handle complex nested structures and text
        result = []
        
        # Extract numeric values
        for key, value in input_data.items():
            if isinstance(value, (int, float)):
                result.append(float(value))
            elif isinstance(value, dict):
                # For dictionaries like emotional states, extract values
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        result.append(float(subvalue))
        
        # Pad to ensure consistent size
        while len(result) < 32:  # Minimum input size
            result.append(0.0)
        
        return result[:64]  # Limit to maximum size
    
    def _prepare_output_tensor(self, output_data: Dict[str, Any]) -> List[float]:
        """
        Convert output data dictionary to a tensor.
        This is a simplified implementation - in a real system, this would
        handle complex output structures properly.
        
        Args:
            output_data: Dictionary of output data
            
        Returns:
            List of floats representing tensor data
        """
        # Similar simplified implementation as input tensor preparation
        result = []
        
        # Extract numeric values
        for key, value in output_data.items():
            if isinstance(value, (int, float)):
                result.append(float(value))
            elif isinstance(value, dict):
                # For dictionaries like emotional states, extract values
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float)):
                        result.append(float(subvalue))
        
        # Pad to ensure consistent size
        while len(result) < 32:  # Minimum output size
            result.append(0.0)
        
        return result[:64]  # Limit to maximum size
    
    def save(self) -> None:
        """Save the entire Mind state, including all neural components."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save mind state
        mind_state_path = self.base_path / f"mind_state_{timestamp}.json"
        with open(mind_state_path, "w") as f:
            # Use model_dump() instead of dict() and use custom encoder
            json.dump(self.mind_state.model_dump(), f, indent=2, cls=NumpyEncoder)
        
        # Create special JSON encoder function for components
        def json_encoder(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'model_dump'):
                return obj.model_dump()
            elif hasattr(obj, 'dict'):
                return obj.dict()
            else:
                raise TypeError(f"Type {type(obj)} not serializable")
        
        # Save each component
        for component_name, component in self.components.items():
            component_path = self.models_path / component_name
            component_path.mkdir(parents=True, exist_ok=True)
            
            try:
                component.save(component_path)
            except Exception as e:
                print(f"Error saving {component_name} component: {e}")
                
                # Try to save state manually if component save fails
                try:
                    if hasattr(component, 'state_dict'):
                        torch.save(component.state_dict(), component_path / f"{component_name}_state.pth")
                        print(f"Saved {component_name} state dict manually")
                except Exception as inner_e:
                    print(f"Failed to save {component_name} manually: {inner_e}")
        
        # Save mother interaction history
        try:
            self.mother.save_interaction_history(self.interactions_path)
        except Exception as e:
            print(f"Error saving mother interaction history: {e}")
            
            # Try to save manually
            try:
                history_path = self.interactions_path / "interaction_history.json"
                with open(history_path, "w") as f:
                    json.dump([], f, indent=2, cls=NumpyEncoder)
                print("Created empty interaction history file")
            except Exception as inner_e:
                print(f"Failed to create interaction history file: {inner_e}")
        
        # Save LMM model (the integration of all components)
        self._save_lmm(timestamp)
        
        print(f"Mind saved at {timestamp}")
    
    def _save_lmm(self, timestamp: str) -> None:
        """
        Save the Large Mind Model (LMM) which represents the integration of
        all neural components.
        
        Args:
            timestamp: Timestamp string for the filename
        """
        # Create a dictionary of all component state dictionaries
        lmm_state = {}
        
        for component_name, component in self.components.items():
            if isinstance(component, NeuralComponent):
                lmm_state[component_name] = component.state_dict()
        
        # Save the LMM state
        lmm_path = self.lmm_path / f"lmm_{timestamp}.pth"
        torch.save(lmm_state, lmm_path)
        
        # Also save the latest version
        latest_path = self.lmm_path / "lmm_latest.pth"
        torch.save(lmm_state, latest_path)
        
        # Save meta-optimizer state
        optimizer_path = self.lmm_path / f"lmm_optimizer_{timestamp}.pth"
        torch.save(self.meta_optimizer.state_dict(), optimizer_path)
        
        # Save latest meta-optimizer state
        latest_optimizer_path = self.lmm_path / "lmm_optimizer_latest.pth"
        torch.save(self.meta_optimizer.state_dict(), latest_optimizer_path)
    
    def load(self) -> None:
        """Load the Mind state, including all neural components."""
        # Find latest mind state file
        mind_state_files = list(self.base_path.glob("mind_state_*.json"))
        if mind_state_files:
            latest_mind_state = max(mind_state_files, key=lambda x: x.stat().st_mtime)
            
            # Load mind state
            with open(latest_mind_state, "r") as f:
                mind_state_data = json.load(f)
                self.mind_state = MindState(**mind_state_data)
        
        # Load each component
        for component_name, component in self.components.items():
            component_path = self.models_path / component_name
            if component_path.exists():
                try:
                    component.load(component_path)
                    print(f"Loaded {component_name} component")
                except Exception as e:
                    print(f"Error loading {component_name} component: {e}")
        
        # Load mother interaction history
        if self.interactions_path.exists():
            self.mother.load_interaction_history(self.interactions_path)
        
        # Load LMM model
        self._load_lmm()
        
        print("Mind loaded successfully")
    
    def _load_lmm(self) -> None:
        """
        Load the Large Mind Model (LMM) which represents the integration of
        all neural components.
        """
        # Check for latest LMM file
        latest_lmm_path = self.lmm_path / "lmm_latest.pth"
        if latest_lmm_path.exists():
            try:
                # Load the LMM state
                lmm_state = torch.load(latest_lmm_path, map_location=self.device)
                
                # Load state into each component
                for component_name, state_dict in lmm_state.items():
                    if component_name in self.components and isinstance(self.components[component_name], NeuralComponent):
                        self.components[component_name].load_state_dict(state_dict)
                
                print("Loaded LMM model")
                
                # Load meta-optimizer state
                latest_optimizer_path = self.lmm_path / "lmm_optimizer_latest.pth"
                if latest_optimizer_path.exists():
                    optimizer_state = torch.load(latest_optimizer_path, map_location=self.device)
                    self.meta_optimizer.load_state_dict(optimizer_state)
                    print("Loaded LMM optimizer state")
            except Exception as e:
                print(f"Error loading LMM model: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Mind.
        
        Returns:
            Dictionary with status information
        """
        # Update mind state first
        self._update_mind_state()
        
        # Basic status information
        status = {
            "age_months": self.mind_state.age_months,
            "developmental_stage": self.mind_state.developmental_stage,
            "vocabulary_size": self.mind_state.vocabulary_size,
            "dominant_emotion": self.mind_state.get_dominant_emotion(),
            "interaction_count": self.interaction_count,
            "component_activations": self.mind_state.component_activations,
            "developmental_metrics": self.mind_state.developmental_metrics
        }
        
        # Add component-specific status
        for component_name, component in self.components.items():
            if component_name == "development":
                status["development_progress"] = self.development_component.get_development_progress()
            elif component_name == "language":
                status["top_words"] = self.language_component.get_top_words(10)
            elif component_name == "memory":
                status["memory_counts"] = self.memory_component.get_memory_counts()
            elif component_name == "social":
                status["attachment_level"] = self.social_component._get_attachment_status("mother").get("attachment_level", 0.0)
                
        return status

if __name__ == "__main__":
    # Test code to ensure the Mind class works
    print("Initializing Neural Child Mind...")
    
    # Create Mind instance
    mind = Mind()
    
    # Print initial status
    print("Initial Mind Status:")
    status = mind.get_status()
    print(json.dumps(status, indent=2))
    
    # Test interaction with mother
    print("\nTesting interaction with Mother...")
    interaction_state = mind.interact_with_mother()
    
    print(f"Mother: {interaction_state.mother_state.get('verbal_response', '')}")
    print(f"Child: {interaction_state.child_state.get('verbal_response', '')}")
    
    # Save the Mind state
    print("\nSaving Mind state...")
    mind.save()
    
    print("Test complete.")
