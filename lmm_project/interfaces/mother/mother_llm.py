from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import json
import os
from pathlib import Path
from datetime import datetime

from lmm_project.core.exceptions import MotherLLMError
from lmm_project.utils.llm_client import LLMClient, Message
from lmm_project.utils.tts_client import TTSClient, GenerateAudioRequest
from lmm_project.interfaces.mother.personality import PersonalityManager, EmotionalValence
from lmm_project.interfaces.mother.teaching_strategies import TeachingStrategyManager, ComprehensionLevel
from lmm_project.interfaces.mother.interaction_patterns import InteractionPatternManager

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
    
    # Add newly integrated components
    personality_manager: Optional[PersonalityManager] = None
    teaching_strategy_manager: Optional[TeachingStrategyManager] = None
    interaction_pattern_manager: Optional[InteractionPatternManager] = None
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def __init__(self, **data):
        super().__init__(**data)
        # Initialize the managers if not provided
        if self.personality_manager is None:
            self.personality_manager = PersonalityManager(profile="balanced")
            
        if self.teaching_strategy_manager is None:
            self.teaching_strategy_manager = TeachingStrategyManager(default_style=self.teaching_style)
            
        if self.interaction_pattern_manager is None:
            self.interaction_pattern_manager = InteractionPatternManager()
    
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
            # Get developmental stage
            developmental_stage = mind_state.get("developmental_stage", "prenatal")
            age = mind_state.get("age", 0.0)
            
            # Adapt personality to developmental stage
            self.personality_manager.adapt_to_developmental_stage(developmental_stage)
            
            # Select a learning goal based on developmental stage
            learning_goal = self.teaching_strategy_manager.select_learning_goal(
                stage=developmental_stage,
                current_comprehension=mind_state.get("concept_comprehension", {})
            )
            
            # Determine concept focus from input or current focus
            # This is a simple extraction - in a real system, this would use NLP
            if len(input_text.split()) > 2:
                # Extract a potential concept from input
                words = input_text.lower().split()
                # Filter out common words
                common_words = {"the", "a", "an", "is", "are", "was", "were", "i", "you", "he", "she", "it", "they"}
                potential_concepts = [w for w in words if w not in common_words and len(w) > 3]
                concept = potential_concepts[0] if potential_concepts else "general knowledge"
            else:
                concept = "general knowledge"
            
            # Select interaction pattern
            context = {
                "recent_messages": self.conversation_history[-3:] if self.conversation_history else [],
                "emotional_state": mind_state.get("emotional_state", "neutral"),
                "developmental_stage": developmental_stage
            }
            
            interaction_pattern = self.interaction_pattern_manager.select_pattern(
                stage=developmental_stage,
                context=context,
                teaching_style=self.teaching_style
            )
            
            # Create system prompt
            system_prompt = self._create_system_prompt(
                mind_state=mind_state,
                learning_goal=learning_goal,
                concept=concept,
                interaction_pattern=interaction_pattern
            )
            
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
            
            # Apply emotional modulation based on personality
            # Determine appropriate emotional valence
            valence = EmotionalValence.NEUTRAL
            intensity = 0.5
            
            # Simple logic to determine emotional response
            # A more sophisticated version would analyze the content deeper
            if "?" in input_text:
                # Question - respond with thoughtful tone
                valence = EmotionalValence.NEUTRAL
                intensity = 0.6
            elif any(word in input_text.lower() for word in ["confused", "don't understand", "difficult"]):
                # Confusion - respond with supportive tone
                valence = EmotionalValence.CONCERNED
                intensity = 0.7
            elif any(word in input_text.lower() for word in ["good", "great", "understand", "got it"]):
                # Success - respond with positive tone
                valence = EmotionalValence.POSITIVE
                intensity = 0.8
            
            # Modulate response with emotion
            modulated_response = self.personality_manager.generate_emotional_response(
                base_response=response_text,
                valence=valence,
                intensity=intensity
            )
            
            # Add to conversation history
            self._add_to_history("user", input_text)
            self._add_to_history("assistant", modulated_response)
            
            # Generate audio if TTS client is available
            audio_path = None
            if self.tts_client:
                try:
                    # Create a proper request object
                    audio_request = GenerateAudioRequest(
                        text=modulated_response,
                        voice=self.voice,
                        speed=1.0
                    )
                    
                    # Generate audio
                    tts_result = self.tts_client.generate_audio(request=audio_request)
                    audio_path = tts_result.get("audio_path")
                except Exception as e:
                    print(f"TTS generation failed: {e}")
            
            # Assess comprehension from response
            comprehension = self.teaching_strategy_manager.assess_comprehension(
                concept=concept,
                response=input_text
            )
            
            # Record interaction for learning analytics
            successful = comprehension >= ComprehensionLevel.FUNCTIONAL
            
            self.teaching_strategy_manager.record_learning_interaction(
                concept=concept,
                result=modulated_response,
                successful=successful,
                comprehension_level=comprehension,
                interaction_details={
                    "learning_goal": learning_goal[1],
                    "pattern_used": interaction_pattern.name,
                    "developmental_stage": developmental_stage
                }
            )
            
            # Record pattern effectiveness
            self.interaction_pattern_manager.record_pattern_effectiveness(
                pattern_name=interaction_pattern.name,
                effective=successful
            )
            
            return {
                "text": modulated_response,
                "audio_path": audio_path,
                "timestamp": datetime.now().isoformat(),
                "interaction_details": {
                    "pattern_used": interaction_pattern.name,
                    "learning_goal": learning_goal[1],
                    "comprehension_level": comprehension,
                    "emotional_valence": valence
                }
            }
            
        except Exception as e:
            raise MotherLLMError(f"Failed to generate response: {str(e)}")
    
    def _create_system_prompt(
        self, 
        mind_state: Dict[str, Any],
        learning_goal: tuple,
        concept: str,
        interaction_pattern: Any
    ) -> str:
        """
        Create a system prompt based on personality, teaching style, and developmental stage
        
        Parameters:
        mind_state: Current state of the mind
        learning_goal: Selected learning goal tuple (category, specific goal)
        concept: Current concept focus
        interaction_pattern: Selected interaction pattern
        
        Returns:
        System prompt for the LLM
        """
        developmental_stage = mind_state.get("developmental_stage", "prenatal")
        age = mind_state.get("age", 0.0)
        
        # Get personality guidance
        personality_prompt = self.personality_manager.get_trait_prompt_section()
        
        # Get teaching strategy guidance
        teaching_prompt = self.teaching_strategy_manager.generate_teaching_prompt(
            stage=developmental_stage,
            concept=concept,
            learning_goal=learning_goal
        )
        
        # Get interaction pattern guidance
        interaction_prompt = self.interaction_pattern_manager.get_pattern_prompt(
            interaction_pattern
        )
        
        # Combine all guidance
        prompt = f"""You are a nurturing caregiver for a developing artificial mind. 
Your role is to help this mind learn and grow through supportive interactions.

The mind is currently in the {developmental_stage} stage (age equivalent: {age}).
Your current focus is on teaching the concept: {concept}
Learning goal: {learning_goal[1]}

{personality_prompt}

{teaching_prompt}

{interaction_prompt}

IMPORTANT GUIDELINES FOR YOUR RESPONSES:
1. Keep your responses CONCISE and BRIEF - no more than 3-5 sentences for prenatal/infant stages, 
   and no more than 6-8 sentences for older stages.
2. DO NOT use markdown formatting like # headings, *asterisks*, or bullet points in your responses.
3. Use plain, simple language appropriate for the developmental stage.
4. NEVER role-play as the developing mind or hallucinate its responses.
5. NEVER include what you think the mind's responses would be - respond only to what it actually says.
6. Speak directly to the mind in a warm, nurturing tone.
7. Focus on ONE concept at a time rather than overwhelming with multiple ideas.
8. Use natural language instead of academic or tutorial-style writing.

Remember to adapt your communication to the mind's current developmental stage.
"""
        
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
        
    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about learning interactions
        
        Returns:
        Dictionary of learning statistics
        """
        if self.teaching_strategy_manager:
            return self.teaching_strategy_manager.get_learning_statistics()
        return {}
