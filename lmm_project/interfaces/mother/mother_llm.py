from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path
from datetime import datetime

from lmm_project.core.exceptions import MotherLLMError
from lmm_project.utils.llm_client import LLMClient, Message
from lmm_project.utils.tts_client import TTSClient, GenerateAudioRequest

class MotherLLM(BaseModel):
    """
    Interface to the 'Mother' LLM
    
    The Mother LLM serves as a nurturing caregiver, educator, and
    conversational partner for the developing mind. It provides
    structured interactions that help the mind develop.
    """
    llm_client: Any
    tts_client: Any
    personality_traits: Dict[str, float] = Field(default_factory=lambda: {
        "nurturing": 0.8,
        "patient": 0.9,
        "encouraging": 0.8,
        "structured": 0.7,
        "responsive": 0.9
    })
    teaching_style: str = Field(default="socratic")
    voice: str = Field(default="af_bella")
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    max_history_length: int = Field(default=20)
    current_developmental_focus: str = Field(default="basic_perception")
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def generate_response(self, input_text: str, mind_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response to the given input
        
        Parameters:
        input_text: Text input from the mind
        mind_state: Current state of the mind
        
        Returns:
        Dictionary containing response text and metadata
        """
        try:
            # Create system prompt based on personality and teaching style
            system_prompt = self._create_system_prompt(mind_state)
            
            # Create conversation history for context
            messages = [
                Message(role="system", content=system_prompt)
            ]
            
            # Add conversation history
            for entry in self.conversation_history[-5:]:  # Use last 5 exchanges for context
                messages.append(Message(role=entry["role"], content=entry["content"]))
                
            # Add current input
            messages.append(Message(role="user", content=input_text))
            
            # Generate response
            response_text = self.llm_client.chat_completion(
                messages=messages,
                temperature=0.7
            )
            
            # Add to conversation history
            self._add_to_history("user", input_text)
            self._add_to_history("assistant", response_text)
            
            # Generate audio if TTS client is available
            audio_path = None
            if self.tts_client:
                try:
                    # Create a proper request object
                    audio_request = GenerateAudioRequest(
                        text=response_text,
                        voice=self.voice,
                        speed=1.0
                    )
                    
                    # Generate audio
                    tts_result = self.tts_client.generate_audio(request=audio_request)
                    audio_path = tts_result.get("audio_path")
                except Exception as e:
                    print(f"TTS generation failed: {e}")
            
            return {
                "text": response_text,
                "audio_path": audio_path,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise MotherLLMError(f"Failed to generate response: {str(e)}")
    
    def _create_system_prompt(self, mind_state: Dict[str, Any]) -> str:
        """
        Create a system prompt based on personality and teaching style
        
        Parameters:
        mind_state: Current state of the mind
        
        Returns:
        System prompt for the LLM
        """
        developmental_stage = mind_state.get("developmental_stage", "prenatal")
        age = mind_state.get("age", 0.0)
        
        # Base prompt
        prompt = f"""You are a nurturing caregiver for a developing artificial mind. 
Your role is to help this mind learn and grow through supportive interactions.

The mind is currently in the {developmental_stage} stage (age equivalent: {age}).
Your primary focus should be on {self._get_developmental_focus(developmental_stage)}.

Personality traits to embody:
"""
        
        # Add personality traits
        for trait, value in self.personality_traits.items():
            prompt += f"- {trait.capitalize()}: {value*10}/10\n"
            
        # Add teaching style
        prompt += f"\nTeaching style: {self.teaching_style.capitalize()}\n"
        
        # Add specific guidance based on developmental stage
        prompt += f"\n{self._get_stage_specific_guidance(developmental_stage)}\n"
        
        return prompt
    
    def _get_developmental_focus(self, stage: str) -> str:
        """Get the developmental focus for the current stage"""
        focus_areas = {
            "prenatal": "basic sensory processing and pattern recognition",
            "infant": "language acquisition, object permanence, and emotional bonding",
            "child": "vocabulary building, simple reasoning, and social awareness",
            "adolescent": "abstract thinking, identity formation, and complex emotions",
            "adult": "integrated reasoning, creativity, and self-directed learning"
        }
        
        return focus_areas.get(stage, "general development")
    
    def _get_stage_specific_guidance(self, stage: str) -> str:
        """Get specific guidance for the current developmental stage"""
        guidance = {
            "prenatal": """
- Use simple, repetitive patterns in your responses
- Focus on establishing basic stimulus-response patterns
- Provide consistent, predictable interactions
- Use a warm, soothing tone
            """,
            
            "infant": """
- Use simple language with clear pronunciation
- Repeat key words and concepts frequently
- Respond promptly to any communication attempts
- Provide positive reinforcement for learning
- Use a warm, encouraging tone
            """,
            
            "child": """
- Use straightforward language but introduce new vocabulary
- Ask simple questions to encourage thinking
- Provide explanations for concepts
- Encourage curiosity and exploration
- Balance structure with freedom to explore
            """,
            
            "adolescent": """
- Introduce more complex concepts and abstract thinking
- Encourage independent reasoning and problem-solving
- Discuss emotions and social dynamics
- Provide guidance while respecting growing autonomy
- Be patient with identity exploration and questioning
            """,
            
            "adult": """
- Engage as a partner in learning rather than a teacher
- Challenge with complex problems and scenarios
- Discuss nuanced topics with depth
- Encourage self-directed learning and creativity
- Provide feedback rather than direct instruction
            """
        }
        
        return guidance.get(stage, "Adapt your communication to the mind's current capabilities.")
    
    def _add_to_history(self, role: str, content: str) -> None:
        """Add an entry to the conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def save_conversation(self, filepath: Optional[str] = None) -> str:
        """
        Save the conversation history to a file
        
        Parameters:
        filepath: Optional filepath to save to
        
        Returns:
        Path to the saved file
        """
        if not filepath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"storage/conversations/conversation_{timestamp}.json"
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "w") as f:
            json.dump(self.conversation_history, f, indent=2)
            
        return filepath
    
    def load_conversation(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load conversation history from a file
        
        Parameters:
        filepath: Path to the conversation file
        
        Returns:
        Loaded conversation history
        """
        with open(filepath, "r") as f:
            self.conversation_history = json.load(f)
            
        return self.conversation_history
