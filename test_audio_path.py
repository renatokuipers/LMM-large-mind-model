import os
import sys
import logging
import soundfile as sf
import sounddevice as sd
import json

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def play_audio_file(file_path):
    """Play an audio file directly using soundfile and sounddevice."""
    try:
        logger.info(f"Attempting to play audio file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False
            
        # Get file size
        file_size = os.path.getsize(file_path)
        logger.info(f"File exists, size: {file_size} bytes")
        
        # Load the audio file
        data, samplerate = sf.read(file_path)
        logger.info(f"Audio file loaded successfully: sample rate={samplerate}, shape={data.shape}")
        
        # Play the audio
        sd.play(data, samplerate)
        logger.info(f"Audio playback started")
        
        # Wait for playback to finish
        sd.wait()
        logger.info(f"Audio playback completed")
        return True
    except Exception as e:
        logger.error(f"Error playing audio: {e}", exc_info=True)
        return False

def extract_audio_path_from_dict(audio_dict):
    """Extract the audio path from the TTS output dictionary."""
    try:
        logger.info(f"Received audio dictionary: {audio_dict}")
        
        # Check if it's a string that might be a JSON
        if isinstance(audio_dict, str):
            try:
                audio_dict = json.loads(audio_dict)
                logger.info(f"Parsed JSON string into dictionary")
            except json.JSONDecodeError:
                # If it's not JSON but a direct path
                if os.path.exists(audio_dict):
                    logger.info(f"Audio dictionary is a direct file path")
                    return audio_dict
                else:
                    logger.error(f"Could not parse as JSON and not a valid path: {audio_dict}")
                    return None
        
        # Common patterns in the TTS output
        if isinstance(audio_dict, dict):
            # Pattern 1: audio_info -> path
            if 'audio_info' in audio_dict and isinstance(audio_dict['audio_info'], dict) and 'path' in audio_dict['audio_info']:
                return audio_dict['audio_info']['path']
                
            # Pattern 2: direct audio_path
            if 'audio_path' in audio_dict:
                return audio_dict['audio_path']
                
            # Pattern 3: nested in result
            if 'result' in audio_dict and isinstance(audio_dict['result'], dict):
                return extract_audio_path_from_dict(audio_dict['result'])
        
        logger.error(f"Could not extract audio path from dictionary: {audio_dict}")
        return None
    except Exception as e:
        logger.error(f"Error extracting audio path: {e}", exc_info=True)
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_audio_path.py <audio_path_or_json>")
        return
    
    logger.info("Starting audio path test")
    
    input_value = sys.argv[1]
    logger.info(f"Input value: {input_value}")
    
    # Try to determine if input is a path or TTS output dictionary
    if os.path.exists(input_value):
        # Direct file path
        file_path = input_value
        logger.info(f"Input is a direct file path")
    else:
        # Try to parse as TTS output
        try:
            # First try to load from file if the input might be a filename
            if os.path.exists(input_value):
                with open(input_value, 'r') as f:
                    audio_dict = json.load(f)
                    logger.info(f"Loaded JSON from file: {input_value}")
            else:
                # Try to parse as direct JSON
                audio_dict = json.loads(input_value)
                logger.info(f"Parsed input as JSON")
            
            file_path = extract_audio_path_from_dict(audio_dict)
            if not file_path:
                logger.error("Could not extract a valid audio path from the input")
                return
                
        except Exception as e:
            logger.error(f"Input is neither a valid file path nor valid JSON: {e}")
            return
    
    # Try to play the audio file
    success = play_audio_file(file_path)
    if success:
        logger.info("Audio test successful!")
    else:
        logger.error("Audio test failed!")

if __name__ == "__main__":
    main() 