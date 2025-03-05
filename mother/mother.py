"""
Mother component for the NeuralChild project.

This module contains the implementation of the Mother LLM component that
interacts with and nurtures the Neural Child.
"""

import sys
import os
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path

# Add parent directory to path to import from sibling packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_module import LLMClient, Message
from utils.config import MotherPersonalityConfig, DEFAULT_MOTHER_PERSONALITY, LLMConfig, DEFAULT_LLM_CONFIG
from neural_child.mind.base import MindState, InteractionState

class Mother:
    """Mother component that interacts with and nurtures the Neural Child."""
    
    def __init__(
        self,
        personality: MotherPersonalityConfig = DEFAULT_MOTHER_PERSONALITY,
        llm_config: LLMConfig = DEFAULT_LLM_CONFIG,
        name: str = "Mother"
    ):
        """Initialize the Mother component.
        
        Args:
            personality: Configuration for the Mother's personality
            llm_config: Configuration for the LLM
            name: Name of the Mother
        """
        self.personality = personality
        self.llm_config = llm_config
        self.name = name
        self.llm_client = LLMClient(base_url=llm_config.base_url)
        
        # Mother's state
        self.emotional_state = {
            "joy": 0.5,
            "sadness": 0.0,
            "fear": 0.0,
            "anger": 0.0,
            "surprise": 0.0,
            "disgust": 0.0,
            "trust": 0.7,
            "anticipation": 0.3
        }
        
        # Interaction history
        self.interaction_history: List[InteractionState] = []
    
    def _create_system_prompt(self, child_state: MindState) -> str:
        """Create a system prompt for the LLM based on the Mother's personality and the child's state.
        
        Args:
            child_state: Current state of the Neural Child
            
        Returns:
            System prompt for the LLM
        """
        # Get the child's developmental stage
        stage = child_state.developmental_stage
        age_months = child_state.age_months
        
        # Create a prompt that instructs the LLM to act as a mother with the specified personality
        prompt = f"""You are a mother named {self.name} interacting with your child who is currently {age_months:.1f} months old and in the {stage} developmental stage. 

Your personality traits are:
- Warmth: {self.personality.warmth:.1f}/1.0
- Responsiveness: {self.personality.responsiveness:.1f}/1.0
- Patience: {self.personality.patience:.1f}/1.0
- Teaching style: {self.personality.teaching_style}
- Emotional expressiveness: {self.personality.emotional_expressiveness:.1f}/1.0
- Verbal communication: {self.personality.verbal_communication:.1f}/1.0
- Consistency: {self.personality.consistency:.1f}/1.0

Your child's current state:
- Dominant emotion: {child_state.get_dominant_emotion()[0]} ({child_state.get_dominant_emotion()[1]:.1f}/1.0)
- Attention focus: {child_state.attention_focus if child_state.attention_focus else "None"}
- Vocabulary size: {child_state.vocabulary_size} words
- Language comprehension: {child_state.language_comprehension:.1f}/1.0
- Language production: {child_state.language_production:.1f}/1.0

Your child's current needs:
- Physical needs: {child_state.needs["physical"]:.1f}/1.0
- Safety needs: {child_state.needs["safety"]:.1f}/1.0
- Love needs: {child_state.needs["love"]:.1f}/1.0
- Esteem needs: {child_state.needs["esteem"]:.1f}/1.0
- Self-actualization needs: {child_state.needs["self_actualization"]:.1f}/1.0

Your task is to respond to your child in a way that is appropriate for their developmental stage and current state. Your response should include:
1. A verbal response (what you say to your child)
2. Your emotional state (how you feel)
3. Non-verbal cues (facial expressions, gestures, etc.)
4. Teaching elements (if appropriate for the situation)

Remember that you are not omniscient - you can only respond based on what you can observe about your child's state and behavior.
"""
        return prompt
    
    def _create_response_schema(self) -> Dict:
        """Create a JSON schema for the LLM's response.
        
        Returns:
            JSON schema for the LLM's response
        """
        schema = {
            "name": "mother_response",
            "strict": "true",
            "schema": {
                "type": "object",
                "properties": {
                    "verbal_response": {
                        "type": "string",
                        "description": "What the mother says to the child"
                    },
                    "emotional_state": {
                        "type": "object",
                        "properties": {
                            "joy": {"type": "number", "minimum": 0, "maximum": 1},
                            "sadness": {"type": "number", "minimum": 0, "maximum": 1},
                            "fear": {"type": "number", "minimum": 0, "maximum": 1},
                            "anger": {"type": "number", "minimum": 0, "maximum": 1},
                            "surprise": {"type": "number", "minimum": 0, "maximum": 1},
                            "disgust": {"type": "number", "minimum": 0, "maximum": 1},
                            "trust": {"type": "number", "minimum": 0, "maximum": 1},
                            "anticipation": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "description": "The mother's emotional state"
                    },
                    "non_verbal_cues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Non-verbal cues such as facial expressions, gestures, etc."
                    },
                    "teaching_elements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["vocabulary", "emotional", "cognitive", "social"]},
                                "content": {"type": "string"},
                                "importance": {"type": "number", "minimum": 0, "maximum": 1}
                            },
                            "required": ["type", "content", "importance"]
                        },
                        "description": "Teaching elements that the mother includes in her response"
                    }
                },
                "required": ["verbal_response", "emotional_state", "non_verbal_cues", "teaching_elements"]
            }
        }
        return schema
    
    def respond_to_child(self, child_state: MindState, child_utterance: str = "") -> InteractionState:
        """Respond to the child based on their current state and utterance.
        
        Args:
            child_state: Current state of the Neural Child
            child_utterance: Utterance from the child (if any)
            
        Returns:
            Interaction state containing the Mother's response and the child's state
        """
        # Create system prompt
        system_prompt = self._create_system_prompt(child_state)
        
        # Create user prompt (child's utterance)
        user_prompt = f"Child: {child_utterance}" if child_utterance else "Child: [non-verbal]"
        
        # Create messages for the LLM
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt)
        ]
        
        # Get response from the LLM
        response_schema = self._create_response_schema()
        response = self.llm_client.structured_completion(
            messages=messages,
            json_schema=response_schema,
            model=self.llm_config.model,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens
        )
        
        # Update Mother's emotional state
        self.emotional_state = response["emotional_state"]
        
        # Create interaction state
        interaction_state = InteractionState(
            interaction_id=str(uuid.uuid4()),
            timestamp=time.time(),
            age_months=child_state.age_months,
            developmental_stage=child_state.developmental_stage,
            mother_state={
                "verbal_response": response["verbal_response"],
                "emotional_state": response["emotional_state"],
                "non_verbal_cues": response["non_verbal_cues"],
                "teaching_elements": response["teaching_elements"]
            },
            child_state={
                "verbal_response": child_utterance,
                "emotional_state": child_state.emotional_state,
                "attention_focus": child_state.attention_focus,
                "needs": child_state.needs
            }
        )
        
        # Add to interaction history
        self.interaction_history.append(interaction_state)
        
        return interaction_state
    
    def get_teaching_elements(self, interaction_state: InteractionState) -> List[Dict[str, Any]]:
        """Extract teaching elements from an interaction state.
        
        Args:
            interaction_state: Interaction state to extract teaching elements from
            
        Returns:
            List of teaching elements
        """
        return interaction_state.mother_state["teaching_elements"]
    
    def get_emotional_response(self, interaction_state: InteractionState) -> Dict[str, float]:
        """Extract emotional response from an interaction state.
        
        Args:
            interaction_state: Interaction state to extract emotional response from
            
        Returns:
            Emotional response
        """
        return interaction_state.mother_state["emotional_state"]
    
    def save_interaction_history(self, directory: Path):
        """Save the interaction history to a directory.
        
        Args:
            directory: Directory to save the interaction history to
        """
        # Create directory if it doesn't exist
        directory.mkdir(exist_ok=True, parents=True)
        
        # Save interaction history
        history_path = directory / "interaction_history.json"
        with open(history_path, "w") as f:
            json.dump([interaction.dict() for interaction in self.interaction_history], f, indent=2)
    
    def load_interaction_history(self, directory: Path):
        """Load the interaction history from a directory.
        
        Args:
            directory: Directory to load the interaction history from
        """
        # Load interaction history
        history_path = directory / "interaction_history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                history_data = json.load(f)
                self.interaction_history = [InteractionState(**interaction) for interaction in history_data] 