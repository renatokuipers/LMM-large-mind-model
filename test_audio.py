import os
import sys
from pathlib import Path

# Add the project root to sys.path so we can import the audio player
sys.path.append(os.path.abspath("."))

from lmm_project.utils.audio_player import play_audio, get_audio_player
from lmm_project.utils.logging_utils import get_module_logger

logger = get_module_logger("test_audio")

def main():
    """Test audio playback directly."""
    # Check if there's a command-line argument for the audio file
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        # Use a default test file or look for recent TTS files
        kokoro_cache_dir = r"E:\pinokio\api\Kokoro-TTS-Local-v1.0.git\cache\GRADIO_TEMP_DIR"
        
        # Look for WAV files in the kokoro cache directory
        wav_files = []
        if os.path.exists(kokoro_cache_dir):
            for root, dirs, files in os.walk(kokoro_cache_dir):
                for file in files:
                    if file.endswith(".wav"):
                        wav_files.append(os.path.join(root, file))
        
        if wav_files:
            # Sort by modification time (newest first)
            wav_files.sort(key=os.path.getmtime, reverse=True)
            audio_path = wav_files[0]
            print(f"Using most recent WAV file: {audio_path}")
        else:
            print("No WAV files found in the Kokoro cache directory.")
            return
    
    # Check if the file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    print(f"Testing audio playback with file: {audio_path}")
    
    # Check audio device availability
    audio_player = get_audio_player()
    if not audio_player.is_audio_device_available():
        print("Warning: No audio devices available for playback")
    
    # Play the audio (blocking)
    print("Starting playback...")
    result = play_audio(audio_path, blocking=True)
    print(f"Playback result: {'Success' if result else 'Failed'}")

if __name__ == "__main__":
    main() 