# tts_module.py
import os
import json
import time
import tempfile
import shutil
from typing import List, Optional, Dict, Any, Literal, Union
from pathlib import Path
from uuid import uuid4

import requests
from pydantic import BaseModel, Field, field_validator

# For audio playback
import soundfile as sf
import sounddevice as sd

# Output directory for generated audio files
OUTPUT_DIRECTORY = "generated"
DEFAULT_FILENAME = "output_voice.wav"

class GenerateAudioRequest(BaseModel):
    text: str
    voice: str = Field(default="af_nicole")
    speed: float = Field(default=1.0, ge=0.1, le=2.0)

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v

class TTSClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7860"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        try:
            self.session.get(f"{self.base_url}", timeout=5).raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Could not connect to TTS API at {self.base_url}: {e}")
    
    def _wait_for_completion(self, file_path: str, max_wait_time: int = 120) -> bool:
        if not os.path.exists(file_path):
            return False
            
        start_time = time.time()
        last_size = 0
        
        while time.time() - start_time < max_wait_time:
            try:
                current_size = os.path.getsize(file_path)
                if current_size > 0 and current_size == last_size:
                    with open(file_path, 'rb') as f:
                        header = f.read(44)
                        if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                            time.sleep(1.0)
                            return True
                last_size = current_size
            except:
                pass
            
            time.sleep(0.5)
            
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0
    
    def generate_audio(self, request: GenerateAudioRequest, save_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate audio from text using the TTS API
        
        Parameters:
        request: GenerateAudioRequest - The request parameters
        save_to: Optional[str] - Path to save the audio file
        
        Returns:
        Dict containing audio_path and phoneme_sequence
        """
        api_data = [
            request.text,
            request.voice,
            request.speed
        ]
        
        # Use the correct endpoint from your API call example
        endpoint = "/gradio_api/call/generate_first"
        
        response = self.session.post(
            f"{self.base_url}{endpoint}",
            json={"data": api_data},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        
        response_json = response.json()
        event_id = response_json.get("event_id")
        
        if not event_id:
            raise ValueError("No event_id in response")
            
        stream_url = f"{self.base_url}{endpoint}/{event_id}"
        stream_response = self.session.get(stream_url, stream=True)
        stream_response.raise_for_status()
        
        data_content = None
        
        for line in stream_response.iter_lines():
            if not line:
                continue
                
            decoded_line = line.decode('utf-8')
            
            if decoded_line.startswith('data:'):
                data_json = decoded_line[5:].strip()
                try:
                    data_content = json.loads(data_json)
                    break
                except:
                    continue
        
        if not data_content or not isinstance(data_content, list) or len(data_content) < 2:
            raise ValueError(f"Invalid data content: {data_content}")
            
        audio_info = data_content[0] 
        phoneme_sequence = data_content[1]
        
        result = {
            "audio_info": audio_info,
            "phoneme_sequence": phoneme_sequence
        }
        
        if isinstance(audio_info, dict) and 'path' in audio_info:
            audio_path = audio_info['path']
            
            if save_to:
                output_path = save_to
            else:
                temp_dir = tempfile.gettempdir()
                temp_filename = f"tts_audio_{uuid4()}.wav"
                output_path = os.path.join(temp_dir, temp_filename)
            
            if os.path.exists(audio_path):
                self._wait_for_completion(audio_path)
            
            if os.path.exists(audio_path):
                shutil.copy2(audio_path, output_path)
                result["audio_path"] = output_path
            
            elif audio_path.startswith(('http://', 'https://')) or audio_info.get('url'):
                url = audio_path if audio_path.startswith(('http://', 'https://')) else audio_info.get('url')
                
                try:
                    audio_response = self.session.get(url, stream=True)
                    audio_response.raise_for_status()
                    
                    with open(output_path, 'wb') as f:
                        for chunk in audio_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            
                    result["audio_path"] = output_path
                except Exception as e:
                    print(f"Error downloading audio: {e}")
            
            result["audio_path"] = audio_path if os.path.exists(audio_path) else output_path
        
        return result

def get_output_path(filename: Optional[str] = None) -> str:
    """Create output directory and return file path"""
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    if not filename:
        filename = DEFAULT_FILENAME
    
    return os.path.join(OUTPUT_DIRECTORY, filename)

def play_audio(file_path: str):
    """
    Play audio file using sounddevice and soundfile
    
    Parameters:
    file_path: str - Path to the audio file to play
    """
    try:
        # Load audio file
        data, samplerate = sf.read(file_path)
        
        # Play audio
        sd.play(data, samplerate)
        
        # Wait until file is done playing
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")

def text_to_speech(
    text: str, 
    voice: str = "af_nicole", 
    speed: float = 1.0,
    output_path: Optional[str] = None,
    auto_play: bool = True
) -> Dict[str, Any]:
    """
    Convert text to speech using the TTS API
    
    Parameters:
    text: str - The text to convert to speech
    voice: str - The voice to use (e.g. "af_nicole", "af_heart")
    speed: float - The speech speed (0.1 to 2.0)
    output_path: Optional[str] - Path to save the audio file
    auto_play: bool - Whether to automatically play the audio after generation
    
    Returns:
    Dict containing audio_path and phoneme_sequence
    """
    client = TTSClient()
    
    if output_path is None:
        output_path = get_output_path()
    
    request = GenerateAudioRequest(
        text=text,
        voice=voice,
        speed=speed
    )
    
    result = client.generate_audio(request, save_to=output_path)
    
    if auto_play and "audio_path" in result and os.path.exists(result["audio_path"]):
        play_audio(result["audio_path"])
    
    return result

def get_available_voices() -> List[str]:
    """
    Return a list of example voices that we know work with the API
    Note: The actual list may be different based on your TTS backend
    
    Returns:
    List of voice IDs
    """
    return ['af_heart', 'af_bella', 'af_nicole', 'af_aoede', 'af_kore', 'af_sarah', 'af_nova', 'af_sky', 'af_alloy', 'af_jessica', 'af_river', 'am_michael', 'am_fenrir', 'am_puck', 'am_echo', 'am_eric', 'am_liam', 'am_onyx', 'am_santa', 'am_adam', 'bf_emma', 'bf_isabella', 'bf_alice', 'bf_lily', 'bm_george', 'bm_fable', 'bm_lewis', 'bm_daniel', 'pf_dora', 'pm_alex', 'pm_santa']

def tips_for_better_speech():
    """
    Return tips for better speech synthesis
    """
    return """
    Tips for Better Results
    
    Improve Speech Quality:
    - Add punctuation: Proper punctuation helps create natural pauses and intonation
    - Use complete sentences: The model performs better with grammatically complete phrases
    - Try different speeds: Some voices sound more natural at slightly faster or slower speeds
    - Consider voice-content match: Choose voices that match the tone of your content
    
    Handling Special Content:
    - Numbers: Write out numbers as words for better pronunciation of important figures
    - Acronyms: Add periods between letters (like "U.S.A.") or write them out
    - Foreign words: The model handles common foreign words, but may struggle with uncommon ones
    - Technical terms: For domain-specific terminology, test different voices
    
    Performance Tips:
    - For longer texts: Break into smaller chunks for better processing
    """

if __name__ == "__main__":
    # Example usage
    # result = text_to_speech("This is another voice from this local Text-to-Speech model. It's more on the soft and ASMR side.")
    #print(f"Audio saved to: {result['audio_path']}")
    # print(f"Phoneme sequence: {result['phoneme_sequence']}")
    
    # Test with different voice and speed (without auto-play)
    result = text_to_speech(
        #"Hello, let me introduce myself. I am Bella. The mother for the Neural Child project that is currently in development.",
        "Hey Chris, this is pretty fast right? Do you see how fast my voice was generated based on this text?",
        voice="af_bella",
        speed=0.9,
        auto_play=True
    )
    print(f"Audio saved to: {result['audio_path']}")
    
    # You can manually play it later if needed
    # play_audio(result["audio_path"])