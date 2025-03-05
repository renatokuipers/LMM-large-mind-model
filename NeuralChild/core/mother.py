"""
Mother/caregiver module for the NeuralChild system.
Uses LLM to simulate a mother interacting with and nurturing the developing child.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ..llm_module import LLMClient, Message, process_stream
from ..core.child import NeuralChild
from ..models.development_models import DevelopmentalStage
from .. import config

# Configure logging
logger = logging.getLogger(__name__)

class Mother:
    """
    Mother/caregiver module that simulates a mother interacting with the child.
    Acts as a natural mother persona, observing and responding to the child's
    signals and behavior without directly accessing the child's internal state.
    """
    
    def __init__(
        self,
        child: NeuralChild,
        llm_client: Optional[LLMClient] = None,
        name: str = "Mom"
    ):
        """
        Initialize the mother module.
        
        Args:
            child: The NeuralChild instance to interact with
            llm_client: Optional LLM client, creates a new one if not provided
            name: Name of the mother
        """
        self.name = name
        self.child = child
        
        # Initialize LLM client if not provided
        if llm_client is None:
            llm_settings = config.LLM
            self.llm_client = LLMClient(base_url=llm_settings["base_url"])
        else:
            self.llm_client = llm_client
        
        # Load mother personality and interaction parameters
        self.mother_params = config.MOTHER
        
        # Load response schemas
        self.response_schemas = config.MOTHER["response_schemas"]
        
        # Initialize interaction history
        self.interaction_history: List[Dict[str, Any]] = []
        
        # Track when last observation and interaction occurred
        self.last_observation_time = time.time()
        self.last_interaction_time = time.time()
        
        # Observations of the child
        self.current_observation: Dict[str, Any] = {}
        self.observation_history: List[Dict[str, Any]] = []
        
        # Track development
        self.last_developmental_stage = None
        
        logger.info(f"Mother {name} initialized")
    
    def observe_child(self) -> Dict[str, Any]:
        """
        Observe the child's current state.
        Only observes what a real mother would be able to see - not internal state.
        
        Returns:
            Dictionary of observations about the child
        """
        # Get observable state from the child
        observable_state = self.child.get_observable_state()
        
        # Update current observation
        self.current_observation = observable_state
        
        # Track observation time
        self.last_observation_time = time.time()
        
        # Keep a history of observations
        self.observation_history.append({
            "timestamp": datetime.now().isoformat(),
            "observation": observable_state
        })
        
        # Limit history length
        max_history = 10
        if len(self.observation_history) > max_history:
            self.observation_history = self.observation_history[-max_history:]
        
        # Check for developmental stage change
        if (self.last_developmental_stage is None or 
            self.last_developmental_stage != observable_state["developmental_stage"]):
            self.last_developmental_stage = observable_state["developmental_stage"]
            logger.info(f"Child has reached developmental stage: {self.last_developmental_stage}")
        
        return observable_state
    
    def _format_observation_for_llm(self) -> str:
        """
        Format the current observation for the LLM prompt.
        
        Returns:
            Formatted string describing the child's state
        """
        obs = self.current_observation
        
        # Basic information
        result = [
            f"The child is currently {obs['simulated_age_months']} months old and in the {obs['developmental_stage']} developmental stage.",
            f"Facial expression: {obs['facial_expression']}",
            f"Body language: {obs['body_language']}"
        ]
        
        # Add vocalization if present
        if obs.get('vocalizations'):
            result.append(f"The child is vocalizing: {obs['vocalizations']}")
        
        # Add emotional state
        emotions = obs['emotional_state'].get('visible_emotions', {})
        if emotions:
            emotion_str = ", ".join([f"{e} ({v:.1f})" for e, v in emotions.items()])
            result.append(f"Emotional state: {emotion_str}")
            result.append(f"Current mood: {obs['emotional_state']['current_mood']['mood']} " +
                         f"(intensity: {obs['emotional_state']['current_mood']['intensity']:.1f})")
        
        # Add attention information
        result.append(f"The child's attention is {'focused' if obs['attention']['focused'] else 'unfocused'} " +
                     f"with an attention span of approximately {obs['attention']['attention_span_seconds']:.1f} seconds.")
        
        # Add activity level
        result.append(f"Activity level: {obs['activity_level']:.1f} out of 1.0")
        
        # Add language development information
        result.append(f"Language development: vocabulary of {obs['language_development']['vocabulary_size']} words, " +
                     f"mean utterance length: {obs['language_development']['mean_utterance_length']:.1f} words")
        
        return "\n".join(result)
    
    def generate_response(self, response_type: str) -> Dict[str, Any]:
        """
        Generate a response to the child based on current observations.
        Uses LLM to simulate a natural mother persona.
        
        Args:
            response_type: Type of response to generate
            
        Returns:
            Structured response based on the specified schema
        """
        # Make sure we have a current observation
        if not self.current_observation:
            self.observe_child()
        
        # Get the appropriate schema
        if response_type not in self.response_schemas:
            logger.warning(f"Unknown response type: {response_type}, using verbal_interaction instead")
            response_type = "verbal_interaction"
        
        schema = self.response_schemas[response_type]
        
        # Format the observation for the LLM
        observation = self._format_observation_for_llm()
        
        # Format previous interactions for context
        recent_interactions = []
        for interaction in self.interaction_history[-3:]:  # Last 3 interactions
            if "speech" in interaction:
                recent_interactions.append(f"You said: \"{interaction['speech']}\"")
            elif "action" in interaction:
                recent_interactions.append(f"You did: {interaction['action']}")
        
        # Create the prompt
        system_prompt = config.LLM["system_prompt"]
        
        # Create the user prompt with context and instruction
        user_prompt = f"""
# Child's Current State
{observation}

# Recent Interactions
{chr(10).join(recent_interactions) if recent_interactions else "This is your first interaction in a while."}

# Your Task
You are the child's mother. Generate a {response_type} response to the child based on their current state.
Respond as a natural, caring mother would, keeping in mind the child's developmental stage and needs.

Do not explain your reasoning or include meta-commentary.
Only provide the response in valid JSON format matching the required schema.
"""
        
        # Add the specific schema information
        user_prompt += f"\n\n# Required Response Schema\n```json\n{json.dumps(schema, indent=2)}\n```"
        
        # Create the messages
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt)
        ]
        
        # Get the response from the LLM
        try:
            response = self.llm_client.structured_completion(
                messages=messages,
                json_schema=schema,
                model=config.LLM["default_model"],
                temperature=config.LLM["temperature"],
                max_tokens=config.LLM["max_tokens"]
            )
            
            # Add response type
            response["response_type"] = response_type
            
            # Log the response
            logger.debug(f"Generated mother response: {response}")
            
            return response
        
        except Exception as e:
            logger.error(f"Error generating mother response: {str(e)}")
            
            # Return a fallback response
            return self._fallback_response(response_type)
    
    def _fallback_response(self, response_type: str) -> Dict[str, Any]:
        """
        Generate a fallback response if LLM fails.
        
        Args:
            response_type: Type of response that was requested
            
        Returns:
            Simple fallback response
        """
        if response_type == "verbal_interaction":
            return {
                "response_type": "verbal_interaction",
                "speech": "There, there. It's okay.",
                "tone": "soothing"
            }
        elif response_type == "physical_interaction":
            return {
                "response_type": "physical_interaction",
                "action": "gentle pat on back",
                "intensity": 0.5
            }
        elif response_type == "teaching_moment":
            return {
                "response_type": "teaching_moment",
                "concept": "gentle",
                "method": "demonstration",
                "speech": "Let's be gentle."
            }
        else:
            return {
                "response_type": "verbal_interaction",
                "speech": "I'm here with you.",
                "tone": "soothing"
            }
    
    def interact_with_child(self, response_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response and interact with the child.
        
        Args:
            response_type: Optional type of response to generate.
                           If None, chooses based on child's state.
            
        Returns:
            The interaction that was performed
        """
        # Observe the child
        observation = self.observe_child()
        
        # Choose response type if not specified
        if response_type is None:
            response_type = self._choose_response_type(observation)
        
        # Generate the response
        response = self.generate_response(response_type)
        
        # Record the interaction
        self.interaction_history.append(response)
        self.last_interaction_time = time.time()
        
        # Have the child process the interaction
        self.child.process_mother_interaction(response)
        
        return response
    
    def _choose_response_type(self, observation: Dict[str, Any]) -> str:
        """
        Choose an appropriate response type based on the child's state.
        
        Args:
            observation: Current observation of the child
            
        Returns:
            The chosen response type
        """
        # Extract relevant information
        emotions = observation['emotional_state'].get('visible_emotions', {})
        developmental_stage = observation['developmental_stage']
        activity_level = observation['activity_level']
        
        # Check for distress
        is_distressed = any(e in ['sadness', 'fear', 'anger'] and v > 0.6 
                          for e, v in emotions.items())
        
        if is_distressed:
            return "response_to_distress"
        
        # Check for teaching opportunity
        is_curious = activity_level > 0.6 and observation['attention']['focused']
        is_verbal_stage = developmental_stage not in [
            "newborn", "early_infancy", "middle_infancy"
        ]
        
        if is_curious and is_verbal_stage:
            return "teaching_moment"
        
        # Check if physical interaction is needed
        needs_physical = (developmental_stage in ["newborn", "early_infancy", "middle_infancy", "late_infancy"] or 
                        'body_language' in observation and "seeking proximity" in observation['body_language'])
        
        if needs_physical:
            return "physical_interaction"
        
        # Default to verbal interaction
        return "verbal_interaction"
    
    def generate_appropriate_responses(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate a set of appropriate responses across different categories.
        Useful for UI display of options.
        
        Returns:
            Dictionary mapping response types to generated responses
        """
        # Observe the child first
        self.observe_child()
        
        # Generate responses for each type
        responses = {}
        
        for response_type in self.response_schemas.keys():
            responses[response_type] = self.generate_response(response_type)
        
        return responses
    
    def automatic_interaction(self, elapsed_seconds: float) -> Optional[Dict[str, Any]]:
        """
        Automatically interact with the child based on elapsed time and child's state.
        
        Args:
            elapsed_seconds: Time elapsed since last update
            
        Returns:
            The interaction performed, or None if no interaction occurred
        """
        # Determine responsiveness based on mother parameters
        responsiveness = self.mother_params["responsiveness"]
        
        # Observe the child
        observation = self.observe_child()
        
        # Calculate interaction probability
        base_probability = 0.1 * elapsed_seconds
        
        # Increase probability based on child's state
        emotions = observation['emotional_state'].get('visible_emotions', {})
        
        # Higher probability if child is distressed
        distress = sum(v for e, v in emotions.items() 
                     if e in ['sadness', 'fear', 'anger', 'disgust'])
        
        # Increase probability based on distress and mother's responsiveness
        probability = min(1.0, base_probability + (distress * 0.5 * responsiveness))
        
        # Decide whether to interact
        if np.random.random() < probability:
            # Interact with the child
            return self.interact_with_child()
        
        return None
    
    def update(self, elapsed_seconds: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Update the mother module based on elapsed time.
        
        Args:
            elapsed_seconds: Optional time elapsed since last update
                             If None, uses the actual elapsed time
            
        Returns:
            The interaction performed, if any
        """
        # Calculate elapsed time if not provided
        if elapsed_seconds is None:
            current_time = time.time()
            elapsed_seconds_since_observation = current_time - self.last_observation_time
            elapsed_seconds_since_interaction = current_time - self.last_interaction_time
        else:
            elapsed_seconds_since_observation = elapsed_seconds
            elapsed_seconds_since_interaction = elapsed_seconds
        
        # Always observe if it's been a while
        if elapsed_seconds_since_observation > 1.0:
            self.observe_child()
        
        # Consider interacting based on elapsed time
        if elapsed_seconds_since_interaction > 5.0:
            return self.automatic_interaction(elapsed_seconds_since_interaction)
        
        return None