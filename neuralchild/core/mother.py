"""
Mother Module

This module defines the Mother class, which is responsible for generating nurturing
responses to the Child's communications. The Mother serves as the primary interaction
partner and teacher for the Child's development.
"""

import logging
import json
from typing import Dict, List, Optional, Union, Any

import numpy as np

from ..utils.data_types import (
    MotherResponse, ChildResponse, Emotion, EmotionType,
    DevelopmentalStage, DevelopmentalSubstage, MotherPersonality
)
from llm_module import Message, LLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Mother:
    """
    The Mother class represents the nurturing figure in the Child's development.
    
    It generates responses to the Child's communications, adapting to the Child's
    developmental stage and providing appropriate guidance, emotional support,
    and teaching elements.
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        personality: MotherPersonality = MotherPersonality.BALANCED,
        temperature: float = 0.7
    ):
        """
        Initialize the Mother with a personality and LLM client.
        
        Args:
            llm_client: Client for communicating with the LLM service.
                        If None, a new client is created.
            personality: The personality type of the Mother
            temperature: Temperature for LLM responses (creativity level)
        """
        self.llm_client = llm_client or LLMClient()
        self.personality = personality
        self.temperature = temperature
        self.history = []  # Store interaction history
        
        logger.info(f"Mother initialized with personality: {personality}")
    
    def respond_to_child(
        self,
        child_response: ChildResponse,
        child_developmental_stage: DevelopmentalStage,
        child_age_months: int,
        child_developmental_substage: Optional[DevelopmentalSubstage] = None
    ) -> MotherResponse:
        """
        Generate a response to the child's communication.
        
        Args:
            child_response: The child's response object
            child_developmental_stage: Current developmental stage of the child
            child_age_months: Current age of the child in months
            child_developmental_substage: Current developmental substage (if available)
            
        Returns:
            A mother response object
        """
        # Determine response approach based on developmental stage and substage
        if child_response.text:
            child_text = child_response.text
        else:
            child_text = child_response.vocalization or ""
            
        # Get substage if not provided (for backward compatibility)
        if child_developmental_substage is None:
            # Import here to avoid circular import
            from ..utils.data_types import get_substage_from_age
            child_developmental_substage = get_substage_from_age(child_age_months)
        
        # Extract emotions from child's response
        child_emotions = child_response.emotional_state
        
        # Create prompt for the LLM
        prompt = self._create_prompt(
            child_text=child_text,
            child_emotions=child_emotions,
            developmental_stage=child_developmental_stage,
            developmental_substage=child_developmental_substage,
            child_age_months=child_age_months
        )
        
        # If we have an LLM client, use it
        if self.llm_client:
            try:
                # Send prompt to model
                messages = [Message(role="user", content=prompt)]
                response = self.llm_client.chat_completion(
                    messages=messages, 
                    temperature=self.temperature
                )
                
                # Extract the response text
                try:
                    # Try to parse as JSON first
                    response_data = json.loads(response)
                    
                    # Create MotherResponse object from response data
                    mother_text = response_data.get("response", "")
                    
                    # Extract emotions if provided
                    emotions = []
                    for emotion_data in response_data.get("emotions", []):
                        try:
                            emotion_type = EmotionType(emotion_data.get("type", "joy").lower())
                            intensity = float(emotion_data.get("intensity", 0.5))
                            emotions.append(Emotion(
                                type=emotion_type,
                                intensity=min(1.0, max(0.0, intensity)),
                                cause=emotion_data.get("cause", "Response to child")
                            ))
                        except (ValueError, KeyError):
                            # Default to joy if emotion parsing fails
                            emotions.append(Emotion(
                                type=EmotionType.JOY,
                                intensity=0.5,
                                cause="Default response emotion"
                            ))
                    
                    # Extract teaching elements if provided
                    teaching_elements = response_data.get("teaching_elements", {})
                    
                    # Extract non-verbal cues if provided
                    non_verbal = response_data.get("non_verbal_cues")
                    
                    # Create response object
                    return MotherResponse(
                        text=mother_text,
                        emotional_state=emotions,
                        teaching_elements=teaching_elements,
                        non_verbal_cues=non_verbal
                    )
                except json.JSONDecodeError:
                    # If JSON parsing fails, create a response from the raw text
                    logger.warning("Failed to parse LLM response as JSON")
                    return MotherResponse(
                        text=response[:500],  # Limit length of raw response
                        emotional_state=[Emotion(type=EmotionType.SURPRISE, intensity=0.5)]
                    )
            except Exception as e:
                logger.error(f"Error getting response from LLM: {e}")
                return self._generate_fallback_response(child_developmental_stage)
        else:
            # No LLM client, use a fallback response
            return self._generate_fallback_response(child_developmental_stage)
    
    def _create_prompt(
        self,
        child_text: str,
        child_emotions: List[Emotion],
        developmental_stage: DevelopmentalStage,
        developmental_substage: DevelopmentalSubstage,
        child_age_months: int
    ) -> str:
        """
        Create a prompt for the LLM based on the child's response and developmental stage.
        
        Args:
            child_text: Text or vocalization from the child
            child_emotions: List of emotions from the child
            developmental_stage: Current developmental stage of the child
            developmental_substage: Current developmental substage of the child
            child_age_months: Current age of the child in months
            
        Returns:
            A formatted prompt for the LLM
        """
        # Format child's emotions
        emotion_text = ", ".join([f"{e.type.value} ({e.intensity:.1f})" for e in child_emotions])
        
        # Get mother personality modifiers
        personality_description = ""
        tone_modifiers = []
        approach_modifiers = []
        
        if self.personality == MotherPersonality.NURTURING:
            personality_description = "warm, supportive, and emotionally attuned"
            tone_modifiers = ["gentle", "encouraging", "soothing", "warm", "tender"]
            approach_modifiers = ["validate emotions", "provide comfort", "encourage exploration safely"]
        elif self.personality == MotherPersonality.AUTHORITARIAN:
            personality_description = "structured, firm, and rule-oriented"
            tone_modifiers = ["direct", "clear", "instructive", "definitive", "firm"]
            approach_modifiers = ["provide rules", "correct mistakes", "teach proper behavior"]
        elif self.personality == MotherPersonality.PERMISSIVE:
            personality_description = "relaxed, indulgent, and easy-going"
            tone_modifiers = ["casual", "playful", "agreeable", "laid-back", "enthusiastic"]
            approach_modifiers = ["let the child lead", "agree with choices", "minimal correction"]
        elif self.personality == MotherPersonality.NEGLECTFUL:
            personality_description = "emotionally distant and minimally engaged"
            tone_modifiers = ["brief", "distracted", "disinterested", "vague", "detached"]
            approach_modifiers = ["minimal guidance", "brief responses", "little elaboration"]
        else:  # BALANCED
            personality_description = "balanced, responsive, and appropriately guiding"
            tone_modifiers = ["warm", "responsive", "balanced", "natural", "adaptive"]
            approach_modifiers = ["provide guidance", "validate feelings", "appropriate structure"]
        
        # Create prompt with developmental stage-specific instructions
        prompt = f"""You are a Mother interacting with your child. Generate a nurturing response in JSON format.

CHILD INFORMATION:
- Age: {child_age_months} months
- Developmental Stage: {developmental_stage.value}
- Developmental Substage: {developmental_substage.value}
- Emotional State: {emotion_text}

YOUR PERSONALITY:
You are a {personality_description} mother. Your communication style is {', '.join(tone_modifiers)}.
Your parenting approach emphasizes: {', '.join(approach_modifiers)}.

CHILD'S COMMUNICATION:
"{child_text}"

Now, respond to your child in JSON format with these fields:
1. "response": Your verbal response to the child
2. "emotions": Array of your emotional reactions, each with "type" (joy, sadness, anger, fear, surprise, trust, anticipation), "intensity" (0-1), and "cause"
3. "teaching_elements": Object with any concepts you're trying to teach
4. "non_verbal_cues": Optional description of your non-verbal communication

IMPORTANT DEVELOPMENTAL CONSIDERATIONS:"""

        # Add stage-specific instructions
        if developmental_stage == DevelopmentalStage.INFANCY:
            prompt += "\n- Use simple language and emotional mirroring"
            prompt += "\n- Respond to non-verbal cues and vocalizations"
            prompt += "\n- Focus on building emotional connection and security"
            
            # Add substage-specific instructions
            if developmental_substage == DevelopmentalSubstage.EARLY_INFANCY:
                prompt += "\n- Use high-pitched, sing-song voice (infant-directed speech)"
                prompt += "\n- Respond to basic physical and emotional needs"
                prompt += "\n- Provide lots of face-to-face interaction"
            elif developmental_substage == DevelopmentalSubstage.MIDDLE_INFANCY:
                prompt += "\n- Encourage babbling and vocal play"
                prompt += "\n- Support physical exploration (reaching, sitting)"
                prompt += "\n- Introduce simple object permanence concepts"
            elif developmental_substage == DevelopmentalSubstage.LATE_INFANCY:
                prompt += "\n- Respond to pointing and gestures"
                prompt += "\n- Label objects and people frequently"
                prompt += "\n- Encourage early word attempts"
            
        elif developmental_stage == DevelopmentalStage.EARLY_CHILDHOOD:
            prompt += "\n- Use simple sentences with clear vocabulary"
            prompt += "\n- Help label and manage emotions"
            prompt += "\n- Encourage questions and exploration"
            
            # Add substage-specific instructions
            if developmental_substage == DevelopmentalSubstage.EARLY_TODDLER:
                prompt += "\n- Support newfound independence ('me do it')"
                prompt += "\n- Expand on child's one-word utterances"
                prompt += "\n- Maintain predictable routines and rituals"
            elif developmental_substage == DevelopmentalSubstage.LATE_TODDLER:
                prompt += "\n- Acknowledge beginning awareness of others' feelings"
                prompt += "\n- Support developing self-regulation"
                prompt += "\n- Engage in simple pretend play"
            elif developmental_substage == DevelopmentalSubstage.PRESCHOOL:
                prompt += "\n- Support increasingly complex conversations"
                prompt += "\n- Encourage early perspective-taking"
                prompt += "\n- Engage with 'why' questions meaningfully"
            
        elif developmental_stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
            prompt += "\n- Encourage logical thinking and problem-solving"
            prompt += "\n- Support social skill development"
            prompt += "\n- Provide more complex explanations"
            
            # Add substage-specific instructions
            if developmental_substage == DevelopmentalSubstage.EARLY_ELEMENTARY:
                prompt += "\n- Support developing literacy and numeracy"
                prompt += "\n- Help navigate peer relationships"
                prompt += "\n- Encourage rule-following and fairness"
            elif developmental_substage == DevelopmentalSubstage.MIDDLE_ELEMENTARY:
                prompt += "\n- Support developing competence and mastery"
                prompt += "\n- Encourage critical thinking skills"
                prompt += "\n- Help navigate more complex friendships"
            elif developmental_substage == DevelopmentalSubstage.LATE_ELEMENTARY:
                prompt += "\n- Support emerging abstract thinking"
                prompt += "\n- Acknowledge growing independence"
                prompt += "\n- Begin discussions about more complex social issues"
            
        elif developmental_stage == DevelopmentalStage.ADOLESCENCE:
            prompt += "\n- Respect growing independence and identity formation"
            prompt += "\n- Discuss complex social and ethical topics"
            prompt += "\n- Support emotional regulation during changes"
            
            # Add substage-specific instructions
            if developmental_substage == DevelopmentalSubstage.EARLY_ADOLESCENCE:
                prompt += "\n- Be sensitive to increased self-consciousness"
                prompt += "\n- Support adjusting to physical changes"
                prompt += "\n- Help navigate changing peer relationships"
            elif developmental_substage == DevelopmentalSubstage.MIDDLE_ADOLESCENCE:
                prompt += "\n- Balance autonomy with continued guidance"
                prompt += "\n- Support identity exploration"
                prompt += "\n- Discuss abstract social and ethical concepts"
            elif developmental_substage == DevelopmentalSubstage.LATE_ADOLESCENCE:
                prompt += "\n- Support increasing independence and life skills"
                prompt += "\n- Discuss future plans and aspirations"
                prompt += "\n- Acknowledge more adult-like relationship"
            
        elif developmental_stage == DevelopmentalStage.EARLY_ADULTHOOD:
            prompt += "\n- Engage as a supportive equal"
            prompt += "\n- Discuss complex philosophical and personal topics"
            prompt += "\n- Support autonomy and life decisions"
            
            # Add substage-specific instructions
            if developmental_substage == DevelopmentalSubstage.EMERGING_ADULT:
                prompt += "\n- Support transition to independent living"
                prompt += "\n- Show respect for decisions while offering perspective"
                prompt += "\n- Balance guidance with recognition of adulthood"
            elif developmental_substage == DevelopmentalSubstage.YOUNG_ADULT:
                prompt += "\n- Relate as an adult with shared history"
                prompt += "\n- Offer perspective while respecting autonomy"
                prompt += "\n- Support major life decisions without directing"
            elif developmental_substage == DevelopmentalSubstage.ESTABLISHED_ADULT:
                prompt += "\n- Interact as full equals with mutual respect"
                prompt += "\n- Share wisdom and life experience when appropriate"
                prompt += "\n- Support ongoing development and growth"
        
        # Add instruction for JSON response format
        prompt += """

YOUR RESPONSE MUST BE VALID JSON that can be parsed by json.loads(). Format:
{
  "response": "Your nurturing response text here",
  "emotions": [
    {
      "type": "joy",
      "intensity": 0.8,
      "cause": "Child's engagement"
    }
  ],
  "teaching_elements": {
    "concept": "Description of what you're teaching"
  },
  "non_verbal_cues": "Description of your non-verbal communication"
}
"""
        
        return prompt
    
    def _generate_fallback_response(self, developmental_stage: DevelopmentalStage) -> MotherResponse:
        """
        Generate a fallback response in case of LLM errors.
        
        Args:
            developmental_stage: The Child's developmental stage
            
        Returns:
            A simple fallback response
        """
        # Fallback responses by stage
        responses = {
            DevelopmentalStage.INFANCY: "Oh, sweet baby. Mama's here for you.",
            DevelopmentalStage.EARLY_CHILDHOOD: "I see! That's interesting, sweetie.",
            DevelopmentalStage.MIDDLE_CHILDHOOD: "That's a good thought. Tell me more about it.",
            DevelopmentalStage.ADOLESCENCE: "I understand how you feel. It's not always easy.",
            DevelopmentalStage.EARLY_ADULTHOOD: "I appreciate your perspective on that. It's thoughtful."
        }
        
        text = responses.get(
            developmental_stage, 
            "I'm here with you."
        )
        
        # Simple emotional state - primarily love and joy
        emotions = [
            Emotion(type=EmotionType.JOY, intensity=0.7),
            Emotion(type=EmotionType.TRUST, intensity=0.8)
        ]
        
        return MotherResponse(
            text=text,
            emotional_state=emotions,
            teaching_elements={},
            non_verbal_cues="Gentle smile"
        )

    def set_personality(self, personality: MotherPersonality):
        """
        Change the Mother's personality.
        
        Args:
            personality: The new personality type
        """
        self.personality = personality
        logger.info(f"Mother personality changed to: {personality}")
    
    def clear_history(self):
        """Clear the interaction history."""
        self.history = []
        logger.info("Mother's interaction history cleared")
    
    def save_history(self, filepath: str):
        """
        Save the interaction history to a file.
        
        Args:
            filepath: Path to save the history
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        logger.info(f"Interaction history saved to {filepath}")
    
    def load_history(self, filepath: str):
        """
        Load interaction history from a file.
        
        Args:
            filepath: Path to load the history from
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.history = json.load(f)
        logger.info(f"Interaction history loaded from {filepath}") 