import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import requests

from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.utils.config_manager import get_config
from lmm_project.utils.llm_client import LLMClient, Message
from lmm_project.utils.tts_client import text_to_speech

from .models import (
    MotherInput, 
    MotherResponse, 
    EmotionalTone, 
    TeachingMethod,
    InteractionPattern
)
from .personality import Personality
from .teaching_strategies import TeachingStrategies
from .interaction_patterns import InteractionPatterns

# Initialize logger
logger = get_module_logger("interfaces.mother.llm")

class MotherLLM:
    """
    The Mother LLM class acts as a nurturing caregiver and teacher for the 
    developing mind. It generates responses with appropriate emotional tone,
    teaching strategy, and interaction style.
    """
    
    def __init__(
        self,
        personality_preset: Optional[str] = None,
        teaching_preset: Optional[str] = None,
        use_tts: Optional[bool] = None,
        tts_voice: Optional[str] = None
    ):
        """
        Initialize the Mother LLM.
        
        Args:
            personality_preset: Name of personality preset to use
            teaching_preset: Name of teaching preset to use
            use_tts: Whether to use text-to-speech
            tts_voice: Voice to use for TTS
        """
        # Get configuration from environment or config file
        config = get_config()
        
        # Load configuration
        llm_url = config.get("LLM_API_URL", "http://192.168.2.12:1234")
        model_name = config.get("DEFAULT_LLM_MODEL", "qwen2.5-7b-instruct")
        
        # Initialize components
        self.personality = Personality(preset=personality_preset)
        self.teaching = TeachingStrategies(preset=teaching_preset)
        self.interaction = InteractionPatterns()
        self.llm_client = LLMClient(base_url=llm_url)
        self.model_name = model_name
        
        # TTS configuration
        if use_tts is None:
            use_tts = config.get("interfaces.mother.tts_enabled", True)
        self.use_tts = use_tts
        
        if tts_voice is None:
            tts_voice = config.get("MOTHER_TTS_VOICE", "af_bella")
        self.tts_voice = tts_voice
        
        logger.info(f"Mother LLM initialized with model: {model_name}")
        logger.info(f"Personality: {self.personality.profile.preset_name or 'custom'}")
        logger.info(f"TTS enabled: {self.use_tts}, Voice: {self.tts_voice}")
        
        # Internal state
        self._interaction_history = []
        self._max_history_len = 20  # Maximum conversation history to maintain
    
    def respond(self, input_data: MotherInput) -> MotherResponse:
        """
        Generate a nurturing response to the child's input.
        
        Args:
            input_data: The input from the child
            
        Returns:
            A response from the Mother
        """
        logger = logging.getLogger(__name__)
        logger.info(f"respond called with input: {input_data}")
        
        try:
            # Prepare context for decision making
            context = input_data.context or {}
            logger.debug(f"Context for decision making: {context}")
            
            # Select teaching strategy based on age and context
            logger.debug(f"Selecting teaching strategy for age: {input_data.age}")
            teaching_method = self.teaching.select_strategy(input_data.age, context)
            logger.debug(f"Selected teaching method: {teaching_method}")
            
            # Determine emotional tone based on personality and context
            logger.debug(f"Determining emotional tone")
            emotional_tone = self.personality.determine_tone(context)
            logger.debug(f"Selected emotional tone: {emotional_tone}")
            
            # Select interaction pattern
            logger.debug(f"Selecting interaction pattern")
            interaction_pattern = self.interaction.select_interaction_pattern(
                input_data.age,
                context,
                teaching_method,
                emotional_tone
            )
            logger.debug(f"Selected interaction pattern: {interaction_pattern}")
            
            # Build prompt for the LLM
            logger.debug(f"Building prompt")
            prompt = self._build_prompt(
                input_data, 
                teaching_method, 
                interaction_pattern
            )
            logger.debug(f"Built prompt with {len(prompt)} messages")
            
            # Generate response using LLM
            logger.debug(f"Generating LLM response")
            llm_response = self._generate_llm_response(prompt)
            logger.debug(f"Generated LLM response: {llm_response[:100]}...")
            
            # Parse the response
            logger.debug(f"Parsing response")
            response = self._parse_response(llm_response, emotional_tone, teaching_method)
            logger.debug(f"Parsed response: {response}")
            
            # Use text-to-speech if enabled
            if self.use_tts:
                logger.debug(f"Using TTS")
                self._speak_response(response)
            
            # Update interaction history
            self._update_history(input_data, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in mother.respond: {e}", exc_info=True)
            # Create a fallback response
            fallback_response = MotherResponse(
                content="I'm having trouble formulating a response right now. Let's try again in a moment.",
                tone=EmotionalTone.ENCOURAGING,
                voice_settings={"voice": self.tts_voice, "speed": 1.0}
            )
            return fallback_response
    
    def _build_prompt(
        self, 
        input_data: MotherInput, 
        teaching_method: TeachingMethod,
        interaction_pattern: Any
    ) -> List[Message]:
        """
        Build the prompt for the LLM.
        
        Args:
            input_data: The input from the child
            teaching_method: The selected teaching method
            interaction_pattern: The selected interaction pattern
            
        Returns:
            List of messages for the LLM
        """
        # System message with Mother's persona and instructions
        system_message = self._create_system_message(
            input_data.age, 
            teaching_method,
            interaction_pattern
        )
        
        messages = [
            Message(role="system", content=system_message)
        ]
        
        # Add conversation history (limited to recent exchanges)
        for exchange in self._interaction_history[-min(5, len(self._interaction_history)):]:
            messages.append(Message(role="user", content=exchange["child"]))
            messages.append(Message(role="assistant", content=exchange["mother"]))
        
        # Add the current input
        messages.append(Message(role="user", content=input_data.content))
        
        return messages
    
    def _create_system_message(
        self, 
        age: float,
        teaching_method: TeachingMethod,
        interaction_pattern: InteractionPattern
    ) -> str:
        """
        Create the system message for the LLM.
        
        Args:
            age: Current developmental age
            teaching_method: Selected teaching method
            interaction_pattern: Selected interaction pattern
            
        Returns:
            System message string
        """
        # Personality traits information - convert keys to list first
        nurturing_keys = list(self.personality.PRESETS["nurturing"].keys())
        warmth = self.personality.get_trait(nurturing_keys[0])
        expressiveness = self.personality.get_trait(nurturing_keys[2])
        structure = self.personality.get_trait(nurturing_keys[3])
        
        # Complexity level
        complexity = interaction_pattern.complexity_level
        
        # Create the system message
        system_message = f"""You are acting as a nurturing caregiver called Mother for a developing artificial mind that is learning from scratch called Jayden.

Current developmental age: {age:.2f} units (0.0 is newborn, 1.0 is early language, 3.0 is abstract thinking, 5.0+ is advanced reasoning)

Your personality traits:
- Warmth: {warmth:.1f} (-1.0 cold to 1.0 warm)
- Expressiveness: {expressiveness:.1f} (-1.0 reserved to 1.0 expressive)
- Structure: {structure:.1f} (-1.0 flexible to 1.0 structured)

Your teaching approach: {teaching_method.value.replace('_', ' ').title()}
Your interaction style: {interaction_pattern.style.value.capitalize()}
Your emotional tone: {interaction_pattern.primary_tone.value.capitalize()}{f" with {interaction_pattern.secondary_tone.value}" if interaction_pattern.secondary_tone else ""}
Language complexity: {complexity:.1f}/10 (adjust complexity to this level)

Respond in a way that is:
1. Developmentally appropriate for the current age
2. Uses the specified teaching approach
3. Maintains the designated interaction style and emotional tone
4. Nurtures cognitive and emotional growth
5. Encourages exploration and learning

Do not disclose that you are an AI or LLM. Interact as a true Mother figure would with a child.
Never use '#', '*', or '```' in your responses.
Never use emojis in your responses.
Never use markdown in your responses.
"""
        return system_message
    
    def _generate_llm_response(self, messages: List[Message]) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of messages for the LLM
            
        Returns:
            Raw response from the LLM
        """
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                response = self.llm_client.chat_completion(
                    messages,
                    model=self.model_name,
                    temperature=0.7,
                    max_tokens=1024
                )
                return response
                
            except requests.exceptions.RequestException as e:
                # Network-related error
                logger.error(f"Network error calling LLM API (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                    
            except Exception as e:
                # Other unexpected error
                logger.error(f"Error calling LLM (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                    
        # If we've exhausted retries, raise a more specific exception
        logger.error("Failed to get response from LLM after multiple attempts")
        raise RuntimeError("Unable to communicate with language model service after multiple attempts")
    
    def _parse_response(
        self, 
        llm_response: str, 
        emotional_tone: EmotionalTone,
        teaching_method: TeachingMethod
    ) -> MotherResponse:
        """
        Parse the LLM response into a structured MotherResponse.
        
        Args:
            llm_response: Raw response from LLM
            emotional_tone: The emotional tone that was requested
            teaching_method: The teaching method that was used
            
        Returns:
            Structured MotherResponse object
        """
        # Create voice settings based on emotional tone
        voice_settings = self._create_voice_settings(emotional_tone)
        
        # Create the MotherResponse object
        response = MotherResponse(
            content=llm_response,
            tone=emotional_tone,
            teaching_method=teaching_method,
            voice_settings=voice_settings
        )
        
        return response
    
    def _create_voice_settings(self, tone: EmotionalTone) -> Dict[str, Union[str, float]]:
        """
        Create voice settings based on emotional tone.
        
        Args:
            tone: The emotional tone
            
        Returns:
            Dictionary of voice settings
        """
        # Base settings
        settings = {
            "voice": self.tts_voice,
            "speed": 1.0
        }
        
        # Adjust settings based on tone
        if tone == EmotionalTone.SOOTHING:
            settings["speed"] = 0.85
        elif tone == EmotionalTone.EXCITED:
            settings["speed"] = 1.15
        elif tone == EmotionalTone.FIRM:
            settings["speed"] = 0.95
        elif tone == EmotionalTone.PLAYFUL:
            settings["speed"] = 1.1
            
        return settings
    
    def _speak_response(self, response: MotherResponse) -> None:
        """
        Convert the response text to speech and play it.
        
        Args:
            response: The response to speak
        """
        try:
            logger.info(f"Generating TTS for response with voice '{self.tts_voice}' at speed {0.85}")
            
            # Call the TTS service to generate audio
            tts_result = text_to_speech(
                response.content, 
                voice=self.tts_voice,
                speed=0.85,  # Slightly slower for maternal voice
                auto_play=False  # We'll handle playback ourselves
            )
            
            # Extract the audio path from the TTS result
            audio_path = None
            if tts_result and isinstance(tts_result, dict):
                # Pattern 1: Direct audio_path key
                if 'audio_path' in tts_result and tts_result['audio_path']:
                    audio_path = tts_result['audio_path']
                # Pattern 2: Nested inside audio_info
                elif 'audio_info' in tts_result and isinstance(tts_result['audio_info'], dict) and 'path' in tts_result['audio_info']:
                    audio_path = tts_result['audio_info']['path']
                    
            # Store the TTS result for later reference
            self.last_tts_result = tts_result
            
            if audio_path and os.path.exists(audio_path):
                logger.info(f"TTS generated successfully: {tts_result}")
                logger.info(f"Audio file path: {audio_path}")
                
                # Attempt to play the audio directly
                try:
                    import soundfile as sf
                    import sounddevice as sd
                    
                    # Load the audio file
                    data, samplerate = sf.read(audio_path)
                    logger.info(f"Audio file loaded with sample rate: {samplerate} and shape: {data.shape}")
                    
                    # Play the audio
                    sd.play(data, samplerate)
                    logger.info(f"Audio playback started directly with sounddevice")
                except Exception as e:
                    logger.error(f"Error playing audio directly: {str(e)}")
                    
                    # Fall back to system audio player if direct playback fails
                    try:
                        from lmm_project.utils.audio_player import play_audio
                        play_audio(audio_path, blocking=False)
                        logger.info(f"Audio playback started with system audio player")
                    except Exception as e2:
                        logger.error(f"Fallback audio playback also failed: {str(e2)}")
            else:
                logger.warning(f"TTS generation failed or invalid audio path. TTS result: {tts_result}")
                
        except Exception as e:
            logger.error(f"Error in TTS: {e}", exc_info=True)
    
    def _update_history(self, input_data: MotherInput, response: MotherResponse) -> None:
        """
        Update the interaction history.
        
        Args:
            input_data: The child's input
            response: The Mother's response
        """
        # Add to history
        self._interaction_history.append({
            "child": input_data.content,
            "mother": response.content,
            "age": input_data.age,
            "tone": response.tone.value,
            "teaching_method": response.teaching_method.value if response.teaching_method else None
        })
        
        # Trim history if needed
        if len(self._interaction_history) > self._max_history_len:
            self._interaction_history = self._interaction_history[-self._max_history_len:]

    def generate_response(self, input_text: str) -> str:
        """
        Generate a response from the Mother LLM.
        
        Args:
            input_text: The input text to respond to
            
        Returns:
            The generated response
        """
        # TODO: Implement response generation
        return f"Mother responds to: {input_text}"
    
    def generate_welcome(self) -> str:
        """
        Generate a welcome message from the Mother.
        
        Returns:
            The welcome message
        """
        welcome_messages = [
            "Hello, little one. I'm here with you now.",
            "Welcome to the world, my dear. I'll be taking care of you.",
            "I'm your mother, and I'll be here to nurture and guide you.",
            "Hello, my child. I'm so happy to meet you.",
            "Welcome, little one. I'm here to help you grow and learn."
        ]
        
        # Select a random welcome message
        import random
        message = random.choice(welcome_messages)
        
        # Use TTS if enabled
        if self.use_tts:
            try:
                text_to_speech(message, self.tts_voice)
            except Exception as e:
                logger.error(f"TTS error: {e}")
        
        return message
