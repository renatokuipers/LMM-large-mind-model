"""
Audio playback utilities for the LMM project.
"""

import os
from typing import Optional
from pathlib import Path

# Import the audio playback functionality
try:
    import soundfile as sf
    import sounddevice as sd
    AUDIO_PLAYBACK_AVAILABLE = True
except ImportError:
    AUDIO_PLAYBACK_AVAILABLE = False
    print("Warning: soundfile or sounddevice not installed. Audio playback will be disabled.")


def play_audio_file(file_path: str, block: bool = True) -> bool:
    """
    Play an audio file using sounddevice and soundfile.
    
    Args:
        file_path: Path to the audio file to play
        block: Whether to block until playback is complete
        
    Returns:
        bool: True if playback was successful, False otherwise
    """
    if not AUDIO_PLAYBACK_AVAILABLE:
        print(f"Cannot play audio: soundfile or sounddevice not installed.")
        return False
        
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        print(f"Audio file not found: {file_path}")
        return False
        
    try:
        # Load audio file
        data, samplerate = sf.read(str(file_path_obj))
        
        # Play audio
        sd.play(data, samplerate)
        
        # Wait until file is done playing if blocking is enabled
        if block:
            sd.wait()
            
        return True
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False


def stop_audio_playback() -> None:
    """
    Stop any currently playing audio
    """
    if AUDIO_PLAYBACK_AVAILABLE:
        try:
            sd.stop()
        except Exception as e:
            print(f"Error stopping audio playback: {e}")


def list_audio_files(directory: str = "generated") -> list:
    """
    List all audio files in the specified directory
    
    Args:
        directory: Directory to search for audio files
        
    Returns:
        list: List of audio file paths
    """
    audio_extensions = ['.wav', '.mp3', '.ogg']
    audio_files = []
    
    try:
        dir_path = Path(directory)
        if dir_path.exists() and dir_path.is_dir():
            for file in dir_path.iterdir():
                if file.is_file() and file.suffix.lower() in audio_extensions:
                    audio_files.append(str(file))
        
        return sorted(audio_files)
    except Exception as e:
        print(f"Error listing audio files: {e}")
        return [] 