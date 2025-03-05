"""
Mother component for the NeuralChild project.

This module defines the Mother class, which represents the nurturing figure that
interacts with the Neural Child. The Mother uses the LLMClient to generate
structured responses that help shape the child's development.
"""

import json
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum

from ..llm_module import LLMClient, Message
from ..config import CONFIG


class EmotionalState(str, Enum):
    """Emotional states the mother can express."""
    HAPPY = "happy"
    CONTENT = "content"
    LOVING = "loving"
    PROUD = "proud"
    PATIENT = "patient"
    CONCERNED = "concerned"
    FRUSTRATED = "frustrated"
    STERN = "stern"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    CALM = "calm"
    ENCOURAGING = "encouraging"


class NonVerbalCue(str, Enum):
    """Non-verbal cues the mother can express."""
    SMILE = "smile"
    NOD = "nod"
    FROWN = "frown"
    HUG = "hug"
    POINT = "point"
    WAVE = "wave"
    CLAP = "clap"
    SHAKE_HEAD = "shake_head"
    RAISED_EYEBROWS = "raised_eyebrows"
    LAUGH = "laugh"
    TILT_HEAD = "tilt_head"
    EXTENDED_ARMS = "extended_arms"


class TeachingElement(BaseModel):
    """Represents a teaching element in the mother's response."""
    concept: str
    explanation: str
    examples: Optional[List[str]] = None
    reinforcement: Optional[str] = None


class MotherResponse(BaseModel):
    """Structured response from the mother to the child."""
    verbal_response: str
    emotional_state: EmotionalState
    non_verbal_cues: List[NonVerbalCue] = Field(default_factory=list)
    teaching_elements: Optional[List[TeachingElement]] = None
    reflection: Optional[str] = None  # Mother's thoughts (not directly shown to child)


class ChildInputType(str, Enum):
    """Types of input that can come from the child."""
    BABBLE = "babble"
    SINGLE_WORD = "single_word"
    SIMPLE_PHRASE = "simple_phrase"
    COMPLEX_SENTENCE = "complex_sentence"
    QUESTION = "question"
    EMOTIONAL_EXPRESSION = "emotional_expression"
    NON_VERBAL = "non_verbal"


class ChildInput(BaseModel):
    """Structured input from the child to the mother."""
    content: str
    input_type: ChildInputType
    developmental_stage: str
    emotional_state: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class Mother:
    """
    Represents the mother figure in the interaction.
    
    The mother uses an LLM to generate responses that nurture the child's development.
    Responses are structured to include verbal communication, emotional states,
    non-verbal cues, and teaching elements.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the Mother with an LLM client.
        
        Args:
            llm_client: LLMClient instance, created if not provided
        """
        self.llm_client = llm_client or LLMClient(base_url=CONFIG.llm.base_url)
        self.personality = CONFIG.mother_personality
        self.interaction_history: List[Dict[str, Any]] = []
        self.system_prompt = self._generate_system_prompt()
    
    def _generate_system_prompt(self) -> str:
        """
        Generate a system prompt based on the mother's personality.
        
        Returns:
            System prompt for the LLM
        """
        # Start with base prompt
        prompt = CONFIG.llm.system_prompt
        
        # Add personality traits
        prompt += "\n\nYour personality traits as a mother are:"
        
        # Format personality traits
        traits = [
            f"- Warmth: {self.personality.warmth:.1f}/1.0 (How emotionally warm you are)",
            f"- Responsiveness: {self.personality.responsiveness:.1f}/1.0 (How responsive you are to the child's needs)",
            f"- Consistency: {self.personality.consistency:.1f}/1.0 (How consistent your behavior is)",
            f"- Patience: {self.personality.patience:.1f}/1.0 (How patient you are with the child)",
            f"- Teaching Focus: {self.personality.teaching_focus:.1f}/1.0 (How much you focus on teaching vs emotional support)",
            f"- Verbosity: {self.personality.verbosity:.1f}/1.0 (How verbose your communication is)",
            f"- Structure: {self.personality.structure:.1f}/1.0 (How much structure vs freedom you provide)"
        ]
        
        # Add custom traits if any
        for trait, value in self.personality.custom_traits.items():
            traits.append(f"- {trait}: {value:.1f}/1.0")
        
        prompt += "\n" + "\n".join(traits)
        
        # Add response format instructions
        prompt += "\n\nYou will provide responses in a structured JSON format with these elements:"
        prompt += "\n- verbal_response: What you say to the child"
        prompt += "\n- emotional_state: Your emotional state (happy, content, loving, proud, patient, concerned, frustrated, stern, neutral, excited, calm, encouraging)"
        prompt += "\n- non_verbal_cues: List of non-verbal cues (smile, nod, frown, hug, point, wave, clap, shake_head, raised_eyebrows, laugh, tilt_head, extended_arms)"
        prompt += "\n- teaching_elements: Optional list of teaching elements (each with concept, explanation, examples, reinforcement)"
        prompt += "\n- reflection: Your private thoughts about the interaction (not shown to the child)"
        
        # Add developmental awareness
        prompt += "\n\nYou will adjust your responses based on the child's developmental stage, which will be provided with each input."
        prompt += "\n\nRemember that you are interacting with a developing mind. Your responses will shape the child's language acquisition, emotional development, and cognitive abilities."
        
        return prompt
    
    def respond_to_child(self, child_input: ChildInput) -> MotherResponse:
        """
        Generate a response to the child's input.
        
        Args:
            child_input: Structured input from the child
            
        Returns:
            Structured response from the mother
        """
        # Prepare messages for the LLM
        messages = [
            Message(role="system", content=self.system_prompt),
        ]
        
        # Add relevant interaction history
        # Only include the last 5 interactions to keep context manageable
        for interaction in self.interaction_history[-5:]:
            messages.append(Message(
                role="user", 
                content=f"Child ({interaction['child_input'].input_type}): {interaction['child_input'].content}"
            ))
            messages.append(Message(
                role="assistant",
                content=f"Mother: {interaction['mother_response'].verbal_response}"
            ))
        
        # Add current input
        input_message = f"""
Child Input:
- Content: "{child_input.content}"
- Type: {child_input.input_type}
- Developmental Stage: {child_input.developmental_stage}
- Emotional State: {child_input.emotional_state or "Unknown"}
        """
        
        # Add context if provided
        if child_input.context:
            input_message += "\nContext:\n"
            for key, value in child_input.context.items():
                input_message += f"- {key}: {value}\n"
        
        messages.append(Message(role="user", content=input_message))
        
        # Define the response schema
        response_schema = {
            "type": "object",
            "properties": {
                "verbal_response": {"type": "string"},
                "emotional_state": {
                    "type": "string", 
                    "enum": [e.value for e in EmotionalState]
                },
                "non_verbal_cues": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [c.value for c in NonVerbalCue]
                    }
                },
                "teaching_elements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "concept": {"type": "string"},
                            "explanation": {"type": "string"},
                            "examples": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "reinforcement": {"type": "string"}
                        },
                        "required": ["concept", "explanation"]
                    }
                },
                "reflection": {"type": "string"}
            },
            "required": ["verbal_response", "emotional_state", "non_verbal_cues"]
        }
        
        # Get response from LLM
        try:
            llm_response = self.llm_client.structured_completion(
                messages=messages,
                json_schema=response_schema,
                model=CONFIG.llm.model,
                temperature=CONFIG.llm.temperature,
                max_tokens=CONFIG.llm.max_tokens
            )
            
            # Create structured response
            response = MotherResponse(**llm_response)
            
            # Store interaction in history
            self.interaction_history.append({
                "child_input": child_input,
                "mother_response": response
            })
            
            return response
            
        except Exception as e:
            # Fallback response in case of LLM errors
            print(f"Error generating mother response: {e}")
            return MotherResponse(
                verbal_response="I'm here with you. Let's continue.",
                emotional_state=EmotionalState.CALM,
                non_verbal_cues=[NonVerbalCue.SMILE, NonVerbalCue.NOD],
                reflection="Had trouble generating a proper response due to technical issues."
            )
    
    def get_interaction_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of interactions.
        
        Returns:
            List of interactions
        """
        return self.interaction_history
    
    def update_personality(self, personality_traits: Dict[str, float]) -> None:
        """
        Update the mother's personality traits.
        
        Args:
            personality_traits: Dictionary of traits to update
        """
        # Only update traits that exist
        for key, value in personality_traits.items():
            if hasattr(self.personality, key):
                setattr(self.personality, key, value)
        
        # Regenerate system prompt with new personality
        self.system_prompt = self._generate_system_prompt()