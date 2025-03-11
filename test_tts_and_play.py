import os
import sys
import logging
import soundfile as sf
import sounddevice as sd
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_audio_path(tts_result):
    """Extract audio path from TTS result dictionary."""
    try:
        logger.info(f"Extracting audio path from TTS result: {tts_result}")
        
        # Check if it's a dictionary
        if isinstance(tts_result, dict):
            # Pattern 1: Direct audio_path key
            if 'audio_path' in tts_result and tts_result['audio_path']:
                return tts_result['audio_path']
                
            # Pattern 2: Nested in audio_info
            if 'audio_info' in tts_result and isinstance(tts_result['audio_info'], dict):
                if 'path' in tts_result['audio_info']:
                    return tts_result['audio_info']['path']
        
        logger.error(f"Could not extract audio path from result")
        return None
    except Exception as e:
        logger.error(f"Error extracting audio path: {e}")
        return None

def play_audio_file(file_path):
    """Play an audio file using soundfile and sounddevice."""
    try:
        logger.info(f"Playing audio file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Audio file does not exist: {file_path}")
            return False
            
        # Get file information
        file_size = os.path.getsize(file_path)
        logger.info(f"Audio file exists with size: {file_size} bytes")
        
        # Load the file
        data, samplerate = sf.read(file_path)
        logger.info(f"Audio file loaded: sample rate={samplerate}, shape={data.shape}")
        
        # Play the audio
        sd.play(data, samplerate)
        logger.info(f"Audio playback started")
        
        # Wait for playback to complete
        sd.wait()
        logger.info(f"Audio playback completed")
        
        return True
    except Exception as e:
        logger.error(f"Error playing audio: {e}")
        return False

def test_tts_and_play():
    """Test TTS generation and audio playback."""
    try:
        from lmm_project.utils.tts_client import text_to_speech
        
        # Test text
        test_text = "Hello! This is a test of the text to speech system with direct audio playback."
        
        # Generate TTS
        logger.info(f"Generating TTS for text: '{test_text}'")
        tts_result = text_to_speech(
            text=test_text,
            voice="af_bella",
            speed=0.85,
            auto_play=False  # We'll handle playback ourselves
        )
        
        logger.info(f"TTS result: {tts_result}")
        
        # Extract audio path
        audio_path = extract_audio_path(tts_result)
        if not audio_path:
            logger.error("Failed to extract audio path from TTS result")
            return False
            
        logger.info(f"Extracted audio path: {audio_path}")
        
        # Play the audio
        success = play_audio_file(audio_path)
        return success
    except Exception as e:
        logger.error(f"Error in TTS and play test: {e}", exc_info=True)
        return False

def main():
    logger.info("Starting TTS and audio playback test")
    
    # Run the test
    success = test_tts_and_play()
    
    if success:
        logger.info("TTS and audio playback test completed successfully!")
        return 0
    else:
        logger.error("TTS and audio playback test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 