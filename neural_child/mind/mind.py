"""
Mind for the Neural Child.

This module contains the central Mind class that integrates all components of the
Neural Child's mind, including language, emotion, cognition, memory, development,
and social components.
"""

import time
import json
import uuid
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

import torch

from neural_child.language.language_component import LanguageComponent
from neural_child.emotion.emotional_component import EmotionalComponent
from neural_child.cognition.cognitive_component import CognitiveComponent
from neural_child.memory.memory_component import MemoryComponent
from neural_child.development.development_component import DevelopmentComponent
from neural_child.social.social_component import SocialComponent
from neural_child.mind.base import MindState, InteractionState

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("neural_child.mind")

class Mind:
    """Central Mind class for the Neural Child.
    
    This class integrates all components of the Neural Child's mind and manages
    the flow of information between them.
    """
    
    def __init__(
        self,
        initial_age_months: float = 0.0,
        development_speed: float = 1.0,
        device: str = "cpu",
        embedding_api_url: str = "http://192.168.2.12:1234/v1/embeddings",
        embedding_model: str = "text-embedding-nomic-embed-text-v1.5@q4_k_m",
        save_dir: Path = Path("./saved_states")
    ):
        """Initialize the Mind.
        
        Args:
            initial_age_months: Initial age in months
            development_speed: Speed of development (1.0 = real-time)
            device: Device to run neural components on (cpu or cuda)
            embedding_api_url: URL for the embedding API
            embedding_model: Model to use for embeddings
            save_dir: Directory to save state to
        """
        self.device = device
        self.embedding_api_url = embedding_api_url
        self.embedding_model = embedding_model
        self.save_dir = save_dir
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        logger.info("Initializing Mind components")
        
        # Development component (manages developmental stages)
        self.development = DevelopmentComponent(
            initial_age_months=initial_age_months,
            development_speed=development_speed,
            name="development_component"
        )
        
        # Memory component
        self.memory = MemoryComponent(
            input_size=128,
            hidden_size=256,
            output_size=128,
            name="memory_component",
            embedding_api_url=embedding_api_url,
            embedding_model=embedding_model
        )
        
        # Emotional component
        self.emotion = EmotionalComponent(
            input_size=128,
            hidden_size=256,
            output_size=128,
            name="emotional_component"
        )
        
        # Language component
        self.language = LanguageComponent(
            input_size=128,
            hidden_size=256,
            output_size=128,
            name="language_component"
        )
        
        # Cognitive component
        self.cognition = CognitiveComponent(
            input_size=128,
            hidden_size=256,
            output_size=128,
            learning_rate=0.01,
            device=device,
            name="cognitive_component",
            embedding_api_url=embedding_api_url,
            embedding_model=embedding_model
        )
        
        # Social component
        self.social = SocialComponent(
            input_size=128,
            hidden_size=256,
            output_size=128,
            learning_rate=0.01,
            device=device,
            name="social_component",
            embedding_api_url=embedding_api_url,
            embedding_model=embedding_model
        )
        
        # Needs (physiological and safety needs that influence behavior)
        self.needs = {
            "hunger": 0.0,        # 0.0 = satisfied, 1.0 = extremely hungry
            "tiredness": 0.0,     # 0.0 = rested, 1.0 = extremely tired
            "discomfort": 0.0,    # 0.0 = comfortable, 1.0 = extremely uncomfortable
            "stimulation": 0.5,   # 0.0 = understimulated, 0.5 = optimal, 1.0 = overstimulated
            "social_contact": 0.5  # 0.0 = isolated, 1.0 = overwhelmed
        }
        
        # Attention focus
        self.attention_focus = {
            "object": None,       # Current object of attention
            "person": None,       # Current person of attention
            "activity": None      # Current activity of attention
        }
        
        # Interaction history
        self.interaction_history: List[InteractionState] = []
        
        # State tracking
        self.last_update_time = time.time()
        self.active = True
        
        # Initialize mind state
        self.mind_state = self._create_mind_state()
        
        logger.info("Mind initialized")
    
    def process_mother_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an interaction with the mother.
        
        Args:
            interaction_data: Dictionary containing interaction data
                - utterance: Mother's utterance
                - emotional_state: Mother's emotional state
                - teaching_elements: Optional elements the mother is actively teaching
                - context: Context of the interaction
                
        Returns:
            Dictionary containing the child's response
        """
        # Log the interaction
        logger.info(f"Processing mother interaction: {interaction_data.get('utterance', '')}")
        
        try:
            # Update mind state first
            self._update_mind_state()
            
            # Extract relevant information
            mother_utterance = interaction_data.get("utterance", "")
            mother_emotional_state = interaction_data.get("emotional_state", {})
            teaching_elements = interaction_data.get("teaching_elements", {})
            context = interaction_data.get("context", {})
            
            # Process through social component (social and attachment processing)
            social_data = {
                "agent": "mother",
                "content": mother_utterance,
                "emotional_tone": mother_emotional_state,
                "context": context,
                "age_months": self.mind_state.age_months
            }
            social_response = self.social.process_interaction(social_data)
            
            # Process through emotional component (emotional response)
            emotional_data = {
                "mother_utterance": mother_utterance,
                "mother_emotional_state": mother_emotional_state,
                "context": context,
                "social_response": social_response,
                "developmental_stage": self.mind_state.developmental_stage
            }
            emotional_response = self.emotion.process(emotional_data)
            
            # Calculate complexity based on utterance and context
            complexity = self._calculate_complexity(mother_utterance, teaching_elements, context)
            
            # Process through cognitive component (understanding)
            cognitive_data = {
                "mother_utterance": mother_utterance,
                "emotional_state": emotional_response["emotional_state"],
                "context": context,
                "complexity": complexity
            }
            cognitive_response = self.cognition.process_input(cognitive_data)
            
            # Process through language component (language processing and generation)
            language_data = {
                "mother_utterance": mother_utterance,
                "emotional_state": emotional_response["emotional_state"],
                "cognitive_state": cognitive_response,
                "teaching_elements": teaching_elements,
                "context": context,
                "developmental_stage": self.mind_state.developmental_stage,
                "age_months": self.mind_state.age_months
            }
            language_response = self.language.process(language_data)
            
            # Create experience for memory
            experience = {
                "type": "interaction",
                "agent": "mother",
                "mother_utterance": mother_utterance,
                "mother_emotional_state": mother_emotional_state,
                "child_response": language_response["utterance"],
                "child_emotional_state": emotional_response["emotional_state"],
                "teaching_elements": teaching_elements,
                "context": context,
                "timestamp": time.time(),
                "age_months": self.mind_state.age_months,
                "developmental_stage": self.mind_state.developmental_stage
            }
            
            # Process through memory component (store experience and retrieve relevant memories)
            memory_input = {
                "current_experience": experience,
                "query": {
                    "content": mother_utterance,
                    "agent": "mother",
                    "emotional_state": mother_emotional_state
                },
                "emotional_state": emotional_response["emotional_state"],
                "age_months": self.mind_state.age_months,
                "developmental_stage": self.mind_state.developmental_stage
            }
            memory_response = self.memory.process(memory_input)
            
            # Update mind state after processing
            self.mind_state = self._create_mind_state()
            
            # Update needs based on interaction
            self._update_needs(emotional_response, cognitive_response, language_response)
            
            # Record interaction in history
            interaction_state = InteractionState(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                mother_state={
                    "utterance": mother_utterance,
                    "emotional_state": mother_emotional_state,
                    "teaching_elements": teaching_elements,
                    "context": context
                },
                child_state=self.mind_state,
                learning_outcomes={
                    "language": language_response.get("learning_outcomes", {}),
                    "emotional": emotional_response.get("learning_outcomes", {}),
                    "cognitive": cognitive_response.get("learning_outcomes", {}),
                    "social": social_response.get("learning_outcomes", {})
                }
            )
            self.interaction_history.append(interaction_state)
            
            # Limit history size
            if len(self.interaction_history) > 100:
                self.interaction_history = self.interaction_history[-100:]
            
            # Prepare and return response
            response = {
                "utterance": language_response["utterance"],
                "emotional_state": emotional_response["emotional_state"],
                "understanding_level": cognitive_response["understanding_level"],
                "attention_level": cognitive_response["attention_level"],
                "social_response": social_response,
                "memories_activated": memory_response.get("retrieved_memories", []),
                "developmental_metrics": {
                    "language": self.language.get_language_development_metrics(),
                    "emotional": self.emotion.get_emotional_development_metrics(),
                    "cognitive": self.cognition.get_developmental_metrics(),
                    "social": self.social.get_social_development_metrics(),
                    "memory": self.memory.get_memory_development_metrics(),
                    "overall": development_progress
                },
                "needs": self.needs,
                "age_months": self.mind_state.age_months,
                "developmental_stage": self.mind_state.developmental_stage
            }
            
            # Log the response
            logger.info(f"Child response: {response['utterance']}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing interaction: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return a simplified response in case of error
            return {
                "utterance": "...",
                "emotional_state": {"confusion": 0.8},
                "error": str(e)
            }
    
    def update(self, delta_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Update the mind state based on elapsed time.
        
        Args:
            delta_seconds: Optional seconds to advance time by (if None, uses real elapsed time)
            
        Returns:
            Dictionary containing update results
        """
        # Calculate elapsed time
        current_time = time.time()
        elapsed_seconds = delta_seconds if delta_seconds is not None else (current_time - self.last_update_time)
        self.last_update_time = current_time
        
        # Update mind state
        self._update_mind_state(elapsed_seconds)
        
        # Decay needs over time
        self._decay_needs(elapsed_seconds)
        
        # Return updated mind state
        return {
            "mind_state": self.mind_state,
            "developmental_updates": {
                "age_months": self.mind_state.age_months,
                "developmental_stage": self.mind_state.developmental_stage
            },
            "needs": self.needs
        }
    
    def _update_mind_state(self, elapsed_seconds: float = 0.0):
        """Update the mind state based on elapsed time and component states.
        
        Args:
            elapsed_seconds: Seconds elapsed since last update
        """
        # Create current mind state
        current_state = self._create_mind_state()
        
        # Update development component
        development_updates = self.development.update(vars(current_state))
        
        # Update memory development based on age
        self.memory._update_memory_development(
            development_updates["age_months"],
            development_updates["developmental_stage"]
        )
        
        # Check for stage transitions
        if development_updates.get("stage_changed", False):
            logger.info(f"Developmental stage changed to: {development_updates['developmental_stage']}")
            
            # Create a memory of the milestone
            milestone_memory = {
                "type": "semantic",
                "concept": f"Development: {development_updates['developmental_stage']}",
                "definition": f"I reached the {development_updates['developmental_stage']} stage of development.",
                "related_concepts": ["growth", "development", "milestone"],
                "importance": 0.9,
                "timestamp": time.time()
            }
            
            # Add milestone memory
            self.memory.process({
                "current_experience": milestone_memory,
                "emotional_state": {"joy": 0.8, "interest": 0.9},
                "age_months": development_updates["age_months"],
                "developmental_stage": development_updates["developmental_stage"]
            })
        
        # Process new developmental milestones
        if "new_milestones" in development_updates and development_updates["new_milestones"]:
            for category, milestones in development_updates["new_milestones"].items():
                for milestone in milestones:
                    logger.info(f"New {category} milestone achieved: {milestone}")
                    
                    # Create a memory of the milestone
                    milestone_memory = {
                        "type": "episodic",
                        "description": f"I reached a new milestone: {milestone}",
                        "category": category,
                        "milestone": milestone,
                        "context": {
                            "age_months": development_updates["age_months"],
                            "developmental_stage": development_updates["developmental_stage"]
                        },
                        "importance": 0.8,
                        "timestamp": time.time()
                    }
                    
                    # Add milestone memory
                    self.memory.process({
                        "current_experience": milestone_memory,
                        "emotional_state": {"joy": 0.7, "surprise": 0.5, "interest": 0.8},
                        "age_months": development_updates["age_months"],
                        "developmental_stage": development_updates["developmental_stage"]
                    })
    
    def _create_mind_state(self) -> MindState:
        """Create a mind state based on current component states.
        
        Returns:
            MindState object representing current mind state
        """
        # Get developmental metrics from development component
        development_progress = self.development.get_development_progress()
        
        # Convert attention_focus dictionary to string representation
        attention_focus_str = None
        if self.attention_focus["object"]:
            attention_focus_str = f"Object: {self.attention_focus['object']}"
        elif self.attention_focus["person"]:
            attention_focus_str = f"Person: {self.attention_focus['person']}"
        elif self.attention_focus["activity"]:
            attention_focus_str = f"Activity: {self.attention_focus['activity']}"
        
        # Create mind state
        mind_state = MindState(
            age_months=self.development.age_months,
            developmental_stage=self.development.current_stage,
            emotional_state=self.emotion.get_emotional_state(),
            language_capabilities={
                "receptive": self.language.get_language_development_metrics()["receptive_language"],
                "expressive": self.language.get_language_development_metrics()["expressive_language"],
                "vocabulary_size": len(self.language.vocabulary)
            },
            memory_state={
                "working_memory_capacity": self.memory.working_memory_capacity,
                "long_term_memory_development": self.memory.long_term_memory_development,
                "memory_counts": self.memory.get_memory_counts()
            },
            cognitive_state={
                "attention": self.cognition.cognitive_development["attention"],
                "problem_solving": self.cognition.cognitive_development["problem_solving"],
                "abstract_thinking": self.cognition.cognitive_development["abstract_thinking"]
            },
            social_state={
                "attachment": self.social.social_development["attachment"],
                "social_awareness": self.social.social_development["social_awareness"],
                "empathy": self.social.social_development["empathy"],
                "theory_of_mind": self.social.social_development["theory_of_mind"]
            },
            needs=self.needs,
            attention_focus=attention_focus_str,
            developmental_metrics={
                "language": self.language.get_language_development_metrics(),
                "emotional": self.emotion.get_emotional_development_metrics(),
                "cognitive": self.cognition.get_developmental_metrics(),
                "social": self.social.get_social_development_metrics(),
                "memory": self.memory.get_memory_development_metrics(),
                "overall": development_progress
            }
        )
        
        return mind_state
    
    def _calculate_complexity(
        self, 
        utterance: str, 
        teaching_elements: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> float:
        """Calculate the complexity of an interaction.
        
        Args:
            utterance: Mother's utterance
            teaching_elements: Teaching elements in the interaction
            context: Context of the interaction
            
        Returns:
            Complexity score from 0.0 to 1.0
        """
        # Base complexity on utterance length
        complexity = min(1.0, len(utterance) / 200)
        
        # Add complexity for teaching elements
        if teaching_elements:
            complexity += len(teaching_elements) * 0.1
        
        # Add complexity for contextual factors
        if context.get("complexity", None) is not None:
            complexity = max(complexity, context["complexity"])
        
        if context.get("abstract_concepts", False):
            complexity += 0.2
        
        if context.get("multiple_perspectives", False):
            complexity += 0.15
        
        # Ensure within bounds
        return max(0.1, min(1.0, complexity))
    
    def _update_needs(
        self,
        emotional_response: Dict[str, Any],
        cognitive_response: Dict[str, Any],
        language_response: Dict[str, Any]
    ):
        """Update needs based on interaction responses.
        
        Args:
            emotional_response: Response from emotional component
            cognitive_response: Response from cognitive component
            language_response: Response from language component
        """
        # Increase tiredness based on cognitive load
        self.needs["tiredness"] = min(1.0, self.needs["tiredness"] + cognitive_response["cognitive_load"] * 0.05)
        
        # Update stimulation based on cognitive response
        if cognitive_response["understanding_level"] < 0.2:
            # Understimulation if too simple
            self.needs["stimulation"] = max(0.0, self.needs["stimulation"] - 0.1)
        elif cognitive_response["understanding_level"] > 0.8:
            # Overstimulation if too complex
            self.needs["stimulation"] = min(1.0, self.needs["stimulation"] + 0.1)
        else:
            # Move toward optimal stimulation
            self.needs["stimulation"] = 0.7 * self.needs["stimulation"] + 0.3 * 0.5
        
        # Update social contact based on interaction
        # Speaking with mother decreases need for social contact
        self.needs["social_contact"] = max(0.0, self.needs["social_contact"] - 0.1)
        
        # Emotional state affects needs
        emotional_state = emotional_response["emotional_state"]
        if emotional_state.get("distress", 0) > 0.7 or emotional_state.get("fear", 0) > 0.7:
            # Distress or fear increases discomfort
            self.needs["discomfort"] = min(1.0, self.needs["discomfort"] + 0.2)
    
    def _decay_needs(self, elapsed_seconds: float):
        """Decay needs over time.
        
        Args:
            elapsed_seconds: Seconds elapsed since last update
        """
        # Convert to hours for more intuitive rates
        elapsed_hours = elapsed_seconds / 3600.0
        
        # Hunger increases over time
        self.needs["hunger"] = min(1.0, self.needs["hunger"] + 0.2 * elapsed_hours)
        
        # Tiredness increases over time
        self.needs["tiredness"] = min(1.0, self.needs["tiredness"] + 0.1 * elapsed_hours)
        
        # Social contact need increases over time
        self.needs["social_contact"] = min(1.0, self.needs["social_contact"] + 0.15 * elapsed_hours)
        
        # Discomfort and stimulation move toward neutral values more slowly
        self.needs["discomfort"] = 0.95 * self.needs["discomfort"]
        
        if self.needs["stimulation"] > 0.5:
            # Overstimulation decreases
            self.needs["stimulation"] = max(0.5, self.needs["stimulation"] - 0.05 * elapsed_hours)
        else:
            # Understimulation increases toward boredom
            self.needs["stimulation"] = min(0.5, self.needs["stimulation"] + 0.08 * elapsed_hours)
    
    def simulate_physiological_event(self, event_type: str, intensity: float = 0.5):
        """Simulate a physiological event.
        
        Args:
            event_type: Type of event (feeding, diaper_change, sleep, etc.)
            intensity: Intensity of the event (0.0 to 1.0)
        """
        if event_type == "feeding":
            # Reduce hunger
            self.needs["hunger"] = max(0.0, self.needs["hunger"] - intensity * 0.8)
            
            # Create memory
            feeding_memory = {
                "type": "episodic",
                "description": f"I was fed",
                "intensity": intensity,
                "context": {
                    "age_months": self.mind_state.age_months,
                    "hunger_before": self.needs["hunger"] + intensity * 0.8
                },
                "timestamp": time.time()
            }
            
            # Add to memory
            self.memory.process({
                "current_experience": feeding_memory,
                "emotional_state": {"joy": 0.6, "satisfaction": 0.8},
                "age_months": self.mind_state.age_months,
                "developmental_stage": self.mind_state.developmental_stage
            })
            
        elif event_type == "diaper_change":
            # Reduce discomfort
            self.needs["discomfort"] = max(0.0, self.needs["discomfort"] - intensity * 0.7)
            
        elif event_type == "sleep":
            # Reduce tiredness
            self.needs["tiredness"] = max(0.0, self.needs["tiredness"] - intensity * 0.9)
            
            # Create memory if significant sleep
            if intensity > 0.6:
                sleep_memory = {
                    "type": "episodic",
                    "description": f"I slept deeply",
                    "intensity": intensity,
                    "context": {
                        "age_months": self.mind_state.age_months,
                        "tiredness_before": self.needs["tiredness"] + intensity * 0.9
                    },
                    "timestamp": time.time()
                }
                
                # Add to memory
                self.memory.process({
                    "current_experience": sleep_memory,
                    "emotional_state": {"contentment": 0.7, "calmness": 0.8},
                    "age_months": self.mind_state.age_months,
                    "developmental_stage": self.mind_state.developmental_stage
                })
                
        elif event_type == "play":
            # Adjust stimulation toward optimal
            if self.needs["stimulation"] < 0.5:
                self.needs["stimulation"] = min(0.7, self.needs["stimulation"] + intensity * 0.3)
            
            # Reduce social contact need
            self.needs["social_contact"] = max(0.0, self.needs["social_contact"] - intensity * 0.4)
            
            # Create memory
            play_memory = {
                "type": "episodic",
                "description": f"I played and had fun",
                "intensity": intensity,
                "context": {
                    "age_months": self.mind_state.age_months
                },
                "timestamp": time.time()
            }
            
            # Add to memory
            self.memory.process({
                "current_experience": play_memory,
                "emotional_state": {"joy": 0.8, "interest": 0.7, "excitement": 0.6},
                "age_months": self.mind_state.age_months,
                "developmental_stage": self.mind_state.developmental_stage
            })
    
    def save(self):
        """Save the mind state to disk."""
        try:
            # Create timestamp for the save directory
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_dir = self.save_dir / f"mind_state_{timestamp}"
            save_dir.mkdir(exist_ok=True, parents=True)
            
            # Save each component
            self.development.save(save_dir)
            self.emotion.save(save_dir)
            self.cognition.save(save_dir)
            self.language.save(save_dir)
            self.memory.save(save_dir)
            self.social.save(save_dir)
            
            # Save mind-level state
            mind_state = {
                "needs": self.needs,
                "attention_focus": self.attention_focus,
                "last_update_time": self.last_update_time,
                "active": self.active,
                "interaction_history": [vars(interaction) for interaction in self.interaction_history]
            }
            
            with open(save_dir / "mind_state.json", "w") as f:
                json.dump(mind_state, f, indent=2)
                
            logger.info(f"Mind state saved to {save_dir}")
            return str(save_dir)
            
        except Exception as e:
            logger.error(f"Error saving mind state: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def load(self, directory: Path):
        """Load the mind state from disk.
        
        Args:
            directory: Directory to load state from
            
        Returns:
            Boolean indicating success
        """
        try:
            directory = Path(directory)
            if not directory.exists():
                logger.error(f"Directory {directory} does not exist")
                return False
                
            # Load each component
            self.development.load(directory)
            self.emotion.load(directory)
            self.cognition.load(directory)
            self.language.load(directory)
            self.memory.load(directory)
            self.social.load(directory)
            
            # Load mind-level state
            mind_state_path = directory / "mind_state.json"
            if mind_state_path.exists():
                with open(mind_state_path, "r") as f:
                    mind_state = json.load(f)
                    
                    self.needs = mind_state["needs"]
                    self.attention_focus = mind_state["attention_focus"]
                    self.last_update_time = mind_state["last_update_time"]
                    self.active = mind_state["active"]
                    
                    # Recreate interaction history objects
                    self.interaction_history = [
                        InteractionState(**interaction) for interaction in mind_state["interaction_history"]
                    ]
            
            # Update mind state
            self.mind_state = self._create_mind_state()
            
            logger.info(f"Mind state loaded from {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading mind state: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def get_developmental_summary(self) -> Dict[str, Any]:
        """Get a summary of developmental metrics across all components.
        
        Returns:
            Dictionary containing developmental summary
        """
        # Get current developmental stage configuration
        stage_config = self.development.get_current_stage_config()
        
        # Get age-appropriate expectations
        expectations = self.development.get_age_appropriate_expectations()
        
        # Get current developmental metrics
        metrics = {
            "language": self.language.get_language_development_metrics(),
            "emotional": self.emotion.get_emotional_development_metrics(),
            "cognitive": self.cognition.get_developmental_metrics(),
            "social": self.social.get_social_development_metrics(),
            "memory": self.memory.get_memory_development_metrics()
        }
        
        # Calculate progress relative to expectations
        progress = {}
        for category in metrics:
            if category in expectations:
                category_progress = {}
                for metric, value in metrics[category].items():
                    expected = expectations[category].get(metric, 0.5)
                    relative_progress = value / max(0.01, expected)
                    category_progress[metric] = {
                        "current": value,
                        "expected": expected,
                        "relative_progress": min(2.0, relative_progress)  # Cap at 200%
                    }
                progress[category] = category_progress
        
        # Get overall development progress
        overall_progress = self.development.get_development_progress()
        
        # Get recent milestones
        milestones = {}
        for category, achieved in self.development.milestones_achieved.items():
            milestones[category] = achieved[-5:] if achieved else []
        
        return {
            "age_months": self.mind_state.age_months,
            "developmental_stage": {
                "name": self.mind_state.developmental_stage,
                "description": stage_config.description if stage_config else "",
                "age_range": {
                    "min_months": stage_config.min_age_months if stage_config else 0,
                    "max_months": stage_config.max_age_months if stage_config else 0
                }
            },
            "metrics": metrics,
            "expectations": expectations,
            "progress": progress,
            "overall_progress": overall_progress,
            "milestones": milestones
        }
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of the current mind state.
        
        Returns:
            Dictionary containing state snapshot
        """
        return {
            "mind_state": vars(self.mind_state),
            "developmental_summary": self.get_developmental_summary(),
            "memory_counts": self.memory.get_memory_counts(),
            "emotional_state": self.emotion.get_current_emotional_state(),
            "language_capabilities": {
                "vocabulary_size": len(self.language.vocabulary),
                "top_words": self.language.get_top_words(20),
                "development": self.language.get_language_development_metrics()
            },
            "social_state": {
                "attachment_figures": self.social.attachment_figures,
                "relationships": self.social.relationships,
                "development": self.social.get_social_development_metrics()
            },
            "needs": self.needs
        } 