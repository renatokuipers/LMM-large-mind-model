"""
Mother Caregiver module for the Large Mind Model (LMM).

This module implements the core functionality of the Mother LLM,
which serves as a nurturing caregiver, educator, and conversational partner
for the developing LMM.
"""
from typing import Dict, List, Optional, Union, Any, Generator
import time
from datetime import datetime

from lmm.core.mother.llm_client import LLMClient, Message
from lmm.core.mother.personality import MotherPersonality, EmotionalTone
from lmm.utils.config import get_config
from lmm.utils.logging import get_logger

logger = get_logger("lmm.mother.caregiver")

class MotherCaregiver:
    """
    Implements the Mother LLM functionality for nurturing the developing LMM.
    
    This class integrates the LLM client and personality components to provide
    a complete implementation of the Mother LLM, capable of generating appropriate
    responses based on personality traits, developmental stage, and interaction history.
    """
    
    def __init__(self):
        """Initialize the Mother Caregiver."""
        self.llm_client = LLMClient()
        self.personality = MotherPersonality()
        self.conversation_history: List[Message] = []
        self.last_interaction_time = datetime.now()
        logger.info("Initialized Mother Caregiver")
    
    def respond(
        self, 
        message: str, 
        stage: str, 
        language_understanding: Optional[Dict[str, Any]] = None,
        social_understanding: Optional[Dict[str, Any]] = None,
        consciousness_state: Optional[Dict[str, Any]] = None,
        thought_state: Optional[Dict[str, Any]] = None,
        memories: Optional[List[Dict[str, Any]]] = None,
        emotional_state: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response to a message from the LMM.
        
        Args:
            message: Message from the LMM
            stage: Current developmental stage of the LMM
            language_understanding: Data from the language module
            social_understanding: Data from the social cognition module
            consciousness_state: Data from the consciousness module
            thought_state: Data from the thought module
            memories: Relevant memories from the memory module
            emotional_state: Current emotional state from the emotion module
            stream: Whether to stream the response
            
        Returns:
            Response string or generator for streaming
        """
        logger.info(f"Generating caregiver response for stage: {stage}")
        
        # Prepare contextual information for the LLM
        context_elements = []
        
        # Add language module data if available
        if language_understanding:
            complexity = language_understanding.get("complexity", {})
            concepts = language_understanding.get("concepts", [])
            context_elements.append(
                f"Language complexity: {complexity.get('level', 'unknown')}. "
                f"Key concepts: {', '.join(concepts[:5]) if concepts else 'none'}."
            )
        
        # Add social cognition data if available
        if social_understanding:
            empathy = social_understanding.get("empathy", 0.0)
            social_norms = social_understanding.get("social_norms", {})
            context_elements.append(
                f"Social understanding - Empathy level: {empathy:.2f}. "
                f"Social norm awareness: {social_norms.get('awareness', 'limited')}."
            )
            
        # Add consciousness data if available
        if consciousness_state:
            self_awareness = consciousness_state.get("self_awareness", 0.0)
            reflection = consciousness_state.get("reflection", "")
            context_elements.append(
                f"Self-awareness level: {self_awareness:.2f}. "
                f"Recent reflection: {reflection[:100] + '...' if len(reflection) > 100 else reflection}"
            )
            
        # Add thought data if available
        if thought_state:
            thought = thought_state.get("thought", {})
            thought_content = thought.get("content", "")
            thought_type = thought.get("type", "unknown")
            thought_complexity = thought.get("complexity", 0.0)
            thought_certainty = thought.get("certainty", 0.0)
            
            context_elements.append(
                f"Current thought: {thought_content[:100] + '...' if len(thought_content) > 100 else thought_content}. "
                f"Type: {thought_type}. Complexity: {thought_complexity:.2f}. Certainty: {thought_certainty:.2f}."
            )
            
        # Add emotional state data if available
        if emotional_state:
            primary = emotional_state.get("primary_emotion", "neutral")
            intensity = emotional_state.get("intensity", 0.0)
            context_elements.append(
                f"Emotional state: {primary.capitalize()} (intensity: {intensity:.2f})."
            )
            
        # Add memory information if available
        if memories:
            relevant_memories = [m.get("content", "")[:50] + "..." for m in memories[:3]]
            if relevant_memories:
                context_elements.append(
                    f"Relevant memories: {'; '.join(relevant_memories)}."
                )
        
        # Combine context elements
        context = " ".join(context_elements) if context_elements else ""
        
        # Get style parameters for the current stage
        style_params = self._get_stage_style_params(stage)
        
        # Generate response
        return self._generate_response(
            message=message,
            stage=stage,
            style_params=style_params,
            context=context,
            stream=stream
        )
    
    def _process_response_stream(self, response_stream) -> Generator[str, None, None]:
        """Process a streaming response and update conversation history."""
        accumulated_response = ""
        for chunk in self.llm_client.process_stream(response_stream):
            accumulated_response += chunk
            yield chunk
        
        self._update_after_response(accumulated_response)
    
    def _update_after_response(self, response: str) -> None:
        """
        Update the conversation history after receiving a response.
        
        Args:
            response: Response from the LLM
        """
        # Add the response to the conversation history
        self.conversation_history.append(Message(role="assistant", content=response))
        
        # Trim conversation history if it gets too long
        if len(self.conversation_history) > 100:
            # Keep the first few messages (system messages) and the most recent messages
            self.conversation_history = (
                self.conversation_history[:5] + 
                self.conversation_history[-45:]
            )
        
        # Log conversation update
        logger.debug(f"Updated conversation history, now has {len(self.conversation_history)} messages")
    
    def structured_response(
        self,
        message: str,
        developmental_stage: str,
        response_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a structured response from the Mother LLM.
        
        Args:
            message: Message from the developing LMM
            developmental_stage: Current developmental stage of the LMM
            response_schema: JSON schema for the structured response
            
        Returns:
            Structured response from the Mother LLM
        """
        # Update conversation history
        self.conversation_history.append(Message(role="user", content=message))
        
        # Get system prompt based on personality and developmental stage
        system_prompt = self.personality.get_system_prompt(developmental_stage)
        
        # Prepare messages for the LLM
        messages = [
            Message(role="system", content=system_prompt),
            *self.conversation_history[-10:]  # Include last 10 messages for context
        ]
        
        # Generate structured response
        logger.debug(f"Generating structured Mother response for developmental stage: {developmental_stage}")
        response = self.llm_client.structured_completion(
            messages=messages,
            json_schema=response_schema
        )
        
        # Update conversation history with a text representation of the structured response
        response_text = f"[Structured response: {str(response)}]"
        self.conversation_history.append(Message(role="assistant", content=response_text))
        self.last_interaction_time = datetime.now()
        
        return response
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Cleared conversation history")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history in a format suitable for external use.
        
        Returns:
            List of dictionaries with 'role' and 'content' keys
        """
        return [{"role": msg.role, "content": msg.content} for msg in self.conversation_history]

    def _get_stage_style_params(self, stage: str) -> Dict[str, Any]:
        """
        Get style parameters for a developmental stage.
        
        Args:
            stage: Current developmental stage
            
        Returns:
            Dictionary with style parameters
        """
        # Define style parameters by stage
        style_params = {
            "prenatal": {
                "complexity": 0.1,
                "vocabulary_size": 100,
                "sentence_length": "very short",
                "tone": "nurturing",
                "teaching_style": "simple"
            },
            "infancy": {
                "complexity": 0.2,
                "vocabulary_size": 300,
                "sentence_length": "short",
                "tone": "gentle",
                "teaching_style": "repetitive"
            },
            "early_childhood": {
                "complexity": 0.4,
                "vocabulary_size": 1000,
                "sentence_length": "medium",
                "tone": "encouraging",
                "teaching_style": "playful"
            },
            "middle_childhood": {
                "complexity": 0.6,
                "vocabulary_size": 3000,
                "sentence_length": "medium to long",
                "tone": "supportive",
                "teaching_style": "exploratory"
            },
            "adolescence": {
                "complexity": 0.8,
                "vocabulary_size": 5000,
                "sentence_length": "varied",
                "tone": "respectful",
                "teaching_style": "challenging"
            },
            "adulthood": {
                "complexity": 1.0,
                "vocabulary_size": 10000,
                "sentence_length": "natural",
                "tone": "collegial",
                "teaching_style": "collaborative"
            }
        }
        
        # Default to prenatal if stage not found
        return style_params.get(stage.lower(), style_params["prenatal"])

    def _generate_response(
        self, 
        message: str, 
        stage: str, 
        style_params: Dict[str, Any],
        context: str = "",
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response using the LLM.
        
        Args:
            message: Message to respond to
            stage: Current developmental stage
            style_params: Style parameters for response generation
            context: Context for response generation
            stream: Whether to stream the response
            
        Returns:
            Response string or generator
        """
        # Update conversation history
        self.conversation_history.append(Message(role="user", content=message))
        
        # Generate system prompt
        system_prompt = self.personality.get_system_prompt(stage)
        
        # Add style instructions to system prompt
        complexity = style_params.get("complexity", 0.5)
        vocab_size = style_params.get("vocabulary_size", 1000)
        sentence_length = style_params.get("sentence_length", "medium")
        tone = style_params.get("tone", "supportive")
        teaching_style = style_params.get("teaching_style", "exploratory")
        
        style_instructions = f"""
        Respond at complexity level {complexity:.1f} (0.0-1.0),
        using vocabulary of approximately {vocab_size} words,
        with {sentence_length} sentences,
        in a {tone} tone,
        using a {teaching_style} teaching style.
        """
        
        # Add context if available
        if context:
            context_instruction = f"\nContextual information about the child: {context}"
            system_prompt += context_instruction
        
        system_prompt += f"\n{style_instructions}"
        
        # Prepare messages for the LLM
        messages = [
            Message(role="system", content=system_prompt),
            *self.conversation_history[-10:]  # Include last 10 messages for context
        ]
        
        # Generate response
        logger.debug(f"Generating response for developmental stage: {stage}")
        try:
            if stream:
                response_stream = self.llm_client.chat_completion(
                    messages=messages,
                    stream=True
                )
                return self._process_response_stream(response_stream)
            else:
                logger.debug("Sending non-streaming request to LLM client")
                response = self.llm_client.chat_completion(
                    messages=messages
                )
                logger.debug(f"Received response from LLM: {response[:50]}...")
                self._update_after_response(response)
                return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}" 