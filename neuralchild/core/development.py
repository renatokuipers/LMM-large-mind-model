"""
Development Module

This module defines the Development class, which manages the progression of the
Child through developmental stages and simulates the passage of time.
"""

import os
import time
import logging
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any

import numpy as np

from ..utils.data_types import (
    DevelopmentalStage, DevelopmentalSubstage, ChildState, SystemState, DevelopmentConfig,
    MotherPersonality, InteractionLog, MotherResponse, ChildResponse, 
    get_substage_from_age, STAGE_TO_SUBSTAGES
)
from .child import Child
from .mother import Mother

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Development:
    """
    The Development class manages the Child's progression through developmental stages.
    
    It simulates the passage of time, coordinates interactions between the Mother and Child,
    and tracks the Child's development over time.
    """
    
    def __init__(
        self,
        child: Child,
        mother: Mother,
        config: Optional[DevelopmentConfig] = None,
        system_state: Optional[SystemState] = None
    ):
        """
        Initialize the development system.
        
        Args:
            child: The Child instance
            mother: The Mother instance
            config: Optional development configuration
            system_state: Optional existing system state to resume from
        """
        self.child = child
        self.mother = mother
        self.config = config or DevelopmentConfig()
        
        # Initialize or load system state
        if system_state:
            self.system_state = system_state
        else:
            self.system_state = SystemState(
                child_state=self.child.state,
                development_config=self.config
            )
        
        # Set random seed for reproducibility if specified
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
        
        # Track last interaction and simulated time
        self.last_interaction_time = datetime.now()
        self.simulated_time = datetime.now()
        
        # Initialize age if not already set
        if self.child.state.simulated_age_months == 0:
            self.child.state.simulated_age_months = self.config.start_age_months
        
        logger.info(f"Development system initialized with acceleration factor: {self.config.time_acceleration_factor}")
        logger.info(f"Child's starting age: {self.child.state.simulated_age_months} months")
    
    def update_simulated_time(self):
        """
        Update the simulated time based on real elapsed time.
        This allows the child to age at an accelerated rate.
        """
        current_time = datetime.now()
        
        # Calculate real elapsed time since last interaction
        real_elapsed = (current_time - self.last_interaction_time).total_seconds()
        
        # Apply acceleration factor to get simulated elapsed time
        simulated_elapsed = real_elapsed * self.config.time_acceleration_factor
        
        # Update simulated time
        self.simulated_time += timedelta(seconds=simulated_elapsed)
        
        # Calculate elapsed months and update child's age
        # For testing purposes, ensure even small increments update the age
        elapsed_months = simulated_elapsed / (30 * 24 * 60 * 60)  # Approximate months
        
        # Always update the age, even for small increments (important for tests)
        old_age = self.child.state.simulated_age_months
        self.child.state.simulated_age_months += elapsed_months
        
        # Log age change if a month or more has passed
        whole_months_passed = int(self.child.state.simulated_age_months) - int(old_age)
        if whole_months_passed >= 1:
            logger.info(f"Child is now {int(self.child.state.simulated_age_months)} months old")
        
        self.last_interaction_time = current_time
    
    def simulate_interaction(
        self,
        initial_vocalization: Optional[str] = None,
        initial_text: Optional[str] = None
    ) -> Tuple[ChildResponse, MotherResponse]:
        """
        Simulate an interaction between the Mother and Child.
        
        Args:
            initial_vocalization: Optional initial vocalization (for infants)
            initial_text: Optional initial text (for older children)
            
        Returns:
            Tuple of (child_response, mother_response)
        """
        # Get the current state of the child
        child_state = self.child.state
        
        # Create an initial child response
        initial_child_response = ChildResponse(
            text=initial_text,
            vocalization=initial_vocalization,
            emotional_state=child_state.current_emotional_state,
            attention_focus="mother"
        )
        
        # Get mother's response to the child
        mother_response = self.mother.respond_to_child(
            child_response=initial_child_response,
            child_developmental_stage=child_state.developmental_stage,
            child_age_months=child_state.simulated_age_months,
            child_developmental_substage=child_state.developmental_substage
        )
        
        # Process mother's response through child's mind
        child_response = self.child.process_mother_response(mother_response)
        
        # Calculate developmental effects
        developmental_effects = self._calculate_developmental_effects()
        
        # Apply random factors if enabled
        if self.config.enable_random_factors:
            self._apply_random_developmental_factors()
        
        # Log the interaction
        self.system_state.interaction_history.append(InteractionLog(
            mother_response=mother_response,
            child_response=child_response,
            developmental_effect=developmental_effects
        ))
        
        # Limit interaction history size
        max_history = 100  # Arbitrary limit to prevent memory issues
        if len(self.system_state.interaction_history) > max_history:
            self.system_state.interaction_history = self.system_state.interaction_history[-max_history:]
        
        # Update simulated time
        self.update_simulated_time()
        
        # Check for developmental stage progression
        progression_occurred = self.child.check_stage_progression()
        
        # Ensure component integration is synchronized after interaction
        if progression_occurred:
            # If progression occurred, synchronize integration to the new stage
            self.child.integration.synchronize_development(self.child.state.developmental_stage)
        
        return child_response, mother_response
    
    def _calculate_developmental_effects(self) -> Dict[str, float]:
        """
        Calculate which developmental metrics improved in the last interaction.
        
        Returns:
            Dict mapping metric names to improvement amounts
        """
        # Look at current metrics vs history to see improvements
        effects = {}
        metrics = self.child.state.metrics
        
        for metric_name, history in metrics.history.items():
            if len(history) >= 2:
                # Calculate improvement from previous value
                improvement = history[-1] - history[-2]
                if improvement > 0:
                    effects[metric_name] = improvement
        
        return effects
    
    def _apply_random_developmental_factors(self):
        """
        Apply random developmental factors to simulate natural variation in development.
        Some days the child learns more, some days less.
        """
        # Only apply random factors occasionally (30% chance)
        if random.random() > 0.3:
            return
        
        metrics = self.child.state.metrics
        
        # Randomly select 1-2 metrics to boost
        num_boosts = random.randint(1, 2)
        metric_names = [name for name in vars(metrics).keys() 
                       if not name.startswith("_") and name != "history"]
        
        boost_metrics = random.sample(metric_names, min(num_boosts, len(metric_names)))
        
        for metric_name in boost_metrics:
            # Apply a small random boost
            boost_amount = random.uniform(0.01, 0.03)
            current_value = getattr(metrics, metric_name)
            
            # Don't exceed 1.0
            new_value = min(1.0, current_value + boost_amount)
            setattr(metrics, metric_name, new_value)
            
            if boost_amount > 0.02:  # Only log significant boosts
                logger.info(f"Developmental boost: {metric_name} +{boost_amount:.3f}")
    
    def accelerate_development(self, months: int) -> List[DevelopmentalStage]:
        """
        Accelerate the child's development by simulating many interactions over time.
        
        Args:
            months: Number of months to simulate
            
        Returns:
            List of developmental stages the child progressed through
        """
        logger.info(f"Starting accelerated development for {months} months")
        
        # Record initial state
        initial_stage = self.child.state.developmental_stage
        initial_substage = self.child.state.developmental_substage
        initial_age = self.child.state.simulated_age_months
        
        stages_progressed = [initial_stage]
        substages_progressed = [initial_substage]
        
        # Calculate the number of interactions to simulate
        # More interactions in earlier stages, fewer in later stages
        interactions_per_month = {
            DevelopmentalStage.INFANCY: 90,  # ~3 per day
            DevelopmentalStage.EARLY_CHILDHOOD: 60,  # ~2 per day
            DevelopmentalStage.MIDDLE_CHILDHOOD: 30,  # ~1 per day
            DevelopmentalStage.ADOLESCENCE: 15,  # ~1 every other day
            DevelopmentalStage.EARLY_ADULTHOOD: 10,  # ~1 every three days
        }
        
        # Substage transition topics - specifically designed to help with transitions
        transition_topics = {
            # Infancy transitions
            DevelopmentalSubstage.EARLY_INFANCY: [
                "Let's practice making eye contact!", 
                "Look at these colorful toys!",
                "Time for tummy time to build those muscles"
            ],
            DevelopmentalSubstage.MIDDLE_INFANCY: [
                "Where did the toy go? Is it under the blanket?",
                "Can you say mama?",
                "Let me help you sit up!"
            ],
            DevelopmentalSubstage.LATE_INFANCY: [
                "Let's practice walking together!",
                "Can you point to what you want?",
                "Let's try using a spoon!"
            ],
            
            # Early childhood transitions
            DevelopmentalSubstage.EARLY_TODDLER: [
                "Let's name all the animals!",
                "Can you tell me what happened today?",
                "Let's practice taking turns!"
            ],
            DevelopmentalSubstage.LATE_TODDLER: [
                "How do you think your friend felt when that happened?",
                "Let's build something together with these blocks!",
                "Tell me a story about your drawing!"
            ],
            DevelopmentalSubstage.PRESCHOOL: [
                "Let's talk about our feelings when we're upset",
                "What do you think will happen in the story?",
                "Let's count to 20 together!"
            ],
            
            # Middle childhood transitions
            DevelopmentalSubstage.EARLY_ELEMENTARY: [
                "How would you solve this problem?",
                "Let's talk about the rules of this game",
                "What did you learn at school today?"
            ],
            DevelopmentalSubstage.MIDDLE_ELEMENTARY: [
                "Why do you think the character did that?",
                "How would you feel if that happened to you?",
                "Let's talk about taking responsibility"
            ],
            DevelopmentalSubstage.LATE_ELEMENTARY: [
                "What do you think is fair in this situation?",
                "Let's discuss how different people see things differently",
                "How would you organize this project?"
            ],
            
            # Adolescence transitions
            DevelopmentalSubstage.EARLY_ADOLESCENCE: [
                "What do you value most in a friendship?",
                "How do you handle disagreements with friends?",
                "What are some of your goals for the future?"
            ],
            DevelopmentalSubstage.MIDDLE_ADOLESCENCE: [
                "How has your perspective changed on this issue?",
                "What factors would you consider in making this decision?",
                "How do you think society influences our choices?"
            ],
            DevelopmentalSubstage.LATE_ADOLESCENCE: [
                "How would you evaluate different viewpoints on this topic?",
                "What strategies help you when you're feeling stressed?",
                "How do you see yourself contributing to society?"
            ],
            
            # Early adulthood transitions
            DevelopmentalSubstage.EMERGING_ADULT: [
                "How do you balance your responsibilities and personal needs?",
                "What principles guide your important life decisions?",
                "How do you integrate new perspectives with your existing beliefs?"
            ],
            DevelopmentalSubstage.YOUNG_ADULT: [
                "How has your understanding of yourself evolved over time?",
                "What systems have you developed to manage complex responsibilities?",
                "How do you approach mentoring or guiding others?"
            ]
        }
        
        # Simulate time passing
        simulated_months = 0
        stage_changes = 0
        substage_changes = 0
        
        while simulated_months < months:
            # Determine current stage and interactions for this month
            current_stage = self.child.state.developmental_stage
            current_substage = self.child.state.developmental_substage
            interactions = interactions_per_month.get(current_stage, 30)
            
            # Adjust interactions if in transition
            in_transition = self.child.state.stage_transition is not None
            if in_transition:
                # More interactions during transitions to facilitate development
                interactions = int(interactions * 1.5)
            
            logger.info(f"Simulating month {simulated_months+1}/{months} "
                       f"({current_stage.value}/{current_substage.value})")
            
            # Simulate interactions for this month
            for i in range(interactions):
                # Check if transition has been completed
                if self.child.state.developmental_stage != current_stage or self.child.state.developmental_substage != current_substage:
                    new_stage = self.child.state.developmental_stage
                    new_substage = self.child.state.developmental_substage
                    
                    # Handle stage changes
                    if new_stage != current_stage:
                        logger.info(f"Development accelerated from {current_stage.value} to {new_stage.value}")
                        stages_progressed.append(new_stage)
                        stage_changes += 1
                    
                    # Handle substage changes
                    if new_substage != current_substage:
                        logger.info(f"Development substage advanced from {current_substage.value} to {new_substage.value}")
                        substages_progressed.append(new_substage)
                        substage_changes += 1
                        
                        # After a stage/substage change, simulate integration-focused interactions
                        self._simulate_integration_focused_interactions(new_stage, new_substage)
                    
                    # Update for the next iteration
                    current_stage = new_stage
                    current_substage = new_substage
                    break  # Break out of the interaction loop to recalculate interactions per month
                
                # Generate appropriate conversation
                is_transition_focused = False
                
                # If in transition, occasionally use transition-focused topics
                if in_transition and random.random() < 0.4:
                    is_transition_focused = True
                    if current_substage in transition_topics:
                        topic = random.choice(transition_topics[current_substage])
                        
                        # Target specific developmental areas
                        if "feeling" in topic.lower():
                            active_component = "emotional_component"
                        elif "problem" in topic.lower() or "solve" in topic.lower():
                            active_component = "cognitive_component"
                        elif "friend" in topic.lower() or "together" in topic.lower():
                            active_component = "social_component"
                        elif "remember" in topic.lower() or "happened" in topic.lower():
                            active_component = "memory_system"
                        elif "think" in topic.lower() or "yourself" in topic.lower():
                            active_component = "consciousness_component"
                        else:
                            active_component = None
                        
                        # Ensure the interaction moves through the integration system
                        self.simulate_interaction(initial_text=topic)
                        
                        # Apply extra integration effects for transition-focused topics
                        if active_component:
                            self.child.integration.apply_cross_component_effects(
                                developmental_stage=current_stage,
                                active_component_id=active_component
                            )
                
                # If not using transition-focused topics, use age-appropriate topics
                if not is_transition_focused:
                    if current_stage == DevelopmentalStage.INFANCY:
                        # For infants, use vocalizations
                        vocalization = "goo" if random.random() < 0.7 else random.choice(["bah", "mah", "dah", "pah"])
                        self.simulate_interaction(initial_vocalization=vocalization)
                    else:
                        # For older children, use text
                        topics = [
                            "What are you doing?",
                            "Can you help me?",
                            "I'm hungry",
                            "I feel happy/sad/angry",
                            "Tell me a story",
                            "Why is the sky blue?",
                            "I made something!",
                            "What happens when we sleep?",
                            "I had a dream",
                            "My friend said..."
                        ]
                        # More complex topics for older children
                        if current_stage.value in ["adolescence", "early_adulthood"]:
                            topics.extend([
                                "What's the meaning of life?",
                                "How do relationships work?",
                                "I'm worried about my future",
                                "What do you think about this idea?",
                                "Why do people act that way?",
                                "I'm having a difficult time with..."
                            ])
                        
                        # Pick a topic randomly
                        topic = random.choice(topics)
                        self.simulate_interaction(initial_text=topic)
            
            # Apply integration effects across all components at the end of each month
            self.child.integration.apply_cross_component_effects(
                developmental_stage=self.child.state.developmental_stage,
                active_component_id="development_system"  # Use development_system as the active component
            )
            
            # Increment simulated time for this month
            self.child.state.simulated_age_months += 1
            simulated_months += 1
            
            # Update and check for progression
            self.child.update_developmental_metrics()
            self.child.check_stage_progression()
        
        # Log final results
        final_stage = self.child.state.developmental_stage
        final_substage = self.child.state.developmental_substage
        logger.info(f"Accelerated development complete: {initial_age}mo ({initial_stage.value}/{initial_substage.value}) -> "
                   f"{self.child.state.simulated_age_months}mo ({final_stage.value}/{final_substage.value})")
        logger.info(f"Stage changes: {stage_changes}, Substage changes: {substage_changes}")
        
        # Remove duplicate stages (if a stage was repeated)
        unique_stages = []
        for stage in stages_progressed:
            if not unique_stages or stage != unique_stages[-1]:
                unique_stages.append(stage)
        
        return unique_stages
    
    def _simulate_integration_focused_interactions(self, new_stage: DevelopmentalStage, new_substage: DevelopmentalSubstage):
        """
        Simulate interactions that specifically focus on integrating components.
        
        Args:
            new_stage: The new developmental stage
            new_substage: The new developmental substage
        """
        # Define integration-focused interactions for each stage
        integration_interactions = {
            DevelopmentalStage.INFANCY: {
                # Infancy integration interactions
                DevelopmentalSubstage.EARLY_INFANCY: [
                    "Look at this bright red ball!",  # Visual focus
                    "Listen to the rattle sound!"     # Auditory focus
                ],
                DevelopmentalSubstage.MIDDLE_INFANCY: [
                    "Where did the toy go?",          # Object permanence
                    "Who's that in the mirror?"       # Self-recognition
                ],
                DevelopmentalSubstage.LATE_INFANCY: [
                    "Can you wave bye-bye?",          # Motor + social
                    "Let's name the body parts!"      # Language + body awareness
                ]
            },
            DevelopmentalStage.EARLY_CHILDHOOD: {
                # Early childhood integration interactions
                DevelopmentalSubstage.EARLY_TODDLER: [
                    "Do you remember when we saw the dog yesterday?",  # Memory + language
                    "How do you feel when you're happy?"                # Emotional + language
                ],
                DevelopmentalSubstage.LATE_TODDLER: [
                    "Let's imagine we're at the beach!",                # Imagination + language
                    "What happens when your friend takes your toy?"     # Social + emotional
                ],
                DevelopmentalSubstage.PRESCHOOL: [
                    "Why do you think the character in the story was sad?",  # Emotional + cognitive
                    "Let's take turns building with blocks!"                 # Social + cognitive
                ]
            },
            DevelopmentalStage.MIDDLE_CHILDHOOD: {
                # Middle childhood integration interactions
                DevelopmentalSubstage.EARLY_ELEMENTARY: [
                    "Can you solve this problem using what you learned before?",  # Memory + cognitive
                    "Why do you think you said that?"                             # Language + consciousness
                ],
                DevelopmentalSubstage.MIDDLE_ELEMENTARY: [
                    "Why do you think you feel angry sometimes?",                 # Emotional + cognitive
                    "How would you solve a problem with your friend?"             # Social + cognitive
                ],
                DevelopmentalSubstage.LATE_ELEMENTARY: [
                    "How do other people see things differently than you do?",    # Consciousness + social
                    "How can you use your experience to solve new problems?"      # Memory + cognitive
                ]
            },
            DevelopmentalStage.ADOLESCENCE: {
                # Adolescence integration interactions
                DevelopmentalSubstage.EARLY_ADOLESCENCE: [
                    "What do you think about your own thinking process?",        # Consciousness + cognitive
                    "How do you think others see you?"                           # Social + consciousness
                ],
                DevelopmentalSubstage.MIDDLE_ADOLESCENCE: [
                    "How have your memories shaped who you are?",                # Memory + consciousness
                    "How do your emotions influence your decisions?"             # Emotional + cognitive
                ],
                DevelopmentalSubstage.LATE_ADOLESCENCE: [
                    "How do you balance logical thinking with emotional responses?",  # Cognitive + emotional
                    "How does your understanding of yourself affect your relationships?"  # Consciousness + social
                ]
            },
            DevelopmentalStage.EARLY_ADULTHOOD: {
                # Early adulthood integration interactions
                DevelopmentalSubstage.EMERGING_ADULT: [
                    "How do your past experiences, emotions, and relationships shape your decisions?",  # Full integration
                    "When you face a problem, how do all aspects of your mind work together?"         # Full integration
                ],
                DevelopmentalSubstage.YOUNG_ADULT: [
                    "How has your understanding of yourself changed over time?",                       # Full integration
                    "How do you balance emotional and logical thinking in complex situations?"        # Full integration
                ],
                DevelopmentalSubstage.ESTABLISHED_ADULT: [
                    "How do you integrate new experiences with your existing understanding of the world?",  # Full integration
                    "How do you approach mentoring others based on your own developmental journey?"        # Full integration
                ]
            }
        }
        
        # Get interactions for current stage and substage
        if new_stage in integration_interactions and new_substage in integration_interactions[new_stage]:
            logger.info(f"Simulating integration-focused interactions for {new_stage.value}/{new_substage.value}")
            
            for interaction in integration_interactions[new_stage][new_substage]:
                self.simulate_interaction(initial_text=interaction)
                
                # Explicitly apply integration effects after each interaction
                self.child.integration.apply_cross_component_effects(
                    developmental_stage=new_stage,
                    active_component_id=None  # Let the system determine the active component
                )
    
    def save_system_state(self, filepath: str):
        """
        Save the entire system state to a file.
        
        Args:
            filepath: Path to save the state
        """
        # Update system state with current child state
        self.system_state.child_state = self.child.state
        self.system_state.last_save_time = datetime.now()
        
        # Use Pydantic's json export
        state_json = self.system_state.model_dump_json(indent=2)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(state_json)
            
        logger.info(f"System state saved to {filepath}")
    
    @classmethod
    def load_system_state(
        cls, 
        filepath: str, 
        mother_personality: Optional[MotherPersonality] = None
    ) -> 'Development':
        """
        Load a system state from a file and create a Development instance.
        
        Args:
            filepath: Path to load the state from
            mother_personality: Optional override for mother personality
            
        Returns:
            A new Development instance with the loaded state
        """
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            state_json = f.read()
            
        # Parse JSON to SystemState
        system_state = SystemState.model_validate_json(state_json)
        
        # Create Child instance from child state
        child = Child(initial_state=system_state.child_state)
        
        # Use personality from system state or override
        personality = mother_personality or system_state.development_config.mother_personality
        
        # Create Mother instance
        mother = Mother(personality=personality)
        
        # Create Development instance
        development = cls(
            child=child,
            mother=mother,
            config=system_state.development_config,
            system_state=system_state
        )
        
        logger.info(f"System state loaded from {filepath}")
        logger.info(f"Child age: {child.state.simulated_age_months} months, stage: {child.state.developmental_stage}")
        
        return development 