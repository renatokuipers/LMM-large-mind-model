import os
import wave
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import sounddevice as sd
import soundfile as sf

from lmm_project.core.exceptions import ResourceUnavailableError
from lmm_project.utils.logging_utils import get_module_logger

# Initialize logger
logger = get_module_logger("audio_player")


class AudioPlayer:
    """
    Audio player for handling playback of audio files.
    Designed to work with Windows systems.
    """

    def __init__(self):
        """Initialize the audio player."""
        self.current_stream = None
        self.is_playing = False
        self.stop_requested = False
        self.playback_thread = None
        self.playback_lock = threading.Lock()

    def play(self, file_path: Union[str, Path], blocking: bool = False) -> bool:
        """
        Play an audio file.
        
        Parameters:
        file_path: Path to the audio file
        blocking: If True, block until playback is complete
        
        Returns:
        True if playback started successfully, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"Audio file not found: {file_path}")
                return False
                
            if blocking:
                # Play synchronously
                return self._play_sync(file_path)
            else:
                # Play asynchronously
                return self._play_async(file_path)
                
        except Exception as e:
            logger.error(f"Error playing audio file: {str(e)}")
            return False

    def _play_sync(self, file_path: Path) -> bool:
        """
        Play audio file synchronously (blocking).
        
        Parameters:
        file_path: Path to the audio file
        
        Returns:
        True if playback successful, False otherwise
        """
        try:
            # Stop any current playback
            self.stop()
            
            # Use the generic player for all file types for better compatibility
            self._play_generic_sync(file_path)
            return True
            
        except Exception as e:
            logger.error(f"Error in synchronous playback: {str(e)}")
            return False

    def _play_async(self, file_path: Path) -> bool:
        """
        Play audio file asynchronously (non-blocking).
        
        Parameters:
        file_path: Path to the audio file
        
        Returns:
        True if playback started successfully, False otherwise
        """
        try:
            # Stop any current playback
            self.stop()
            
            # Reset flags
            with self.playback_lock:
                self.is_playing = True
                self.stop_requested = False
            
            # Start playback in a new thread
            self.playback_thread = threading.Thread(
                target=self._playback_worker,
                args=(file_path,),
                daemon=True
            )
            self.playback_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting asynchronous playback: {str(e)}")
            with self.playback_lock:
                self.is_playing = False
            return False

    def _playback_worker(self, file_path: Path) -> None:
        """Background worker for asynchronous playback."""
        try:
            # Use the generic player for all file types for better compatibility
            self._play_generic_sync(file_path)
                
        except Exception as e:
            logger.error(f"Error in playback worker: {str(e)}")
        finally:
            # Clean up
            with self.playback_lock:
                self.is_playing = False
                self.current_stream = None

    def _play_wav_sync(self, file_path: Path) -> None:
        """
        Play WAV file using wave module (better for Windows compatibility).
        This method is kept for backward compatibility but is no longer used.
        
        Parameters:
        file_path: Path to the WAV file
        """
        try:
            # Delegate to the generic player for better format support
            self._play_generic_sync(file_path)
        except Exception as e:
            logger.error(f"Error playing WAV file: {str(e)}")

    def _play_generic_sync(self, file_path: Path) -> None:
        """
        Play audio file using soundfile (supports various formats).
        
        Parameters:
        file_path: Path to the audio file
        """
        try:
            # Load the file
            data, sample_rate = sf.read(file_path)
            logger.info(f"Audio file loaded: {file_path}, sample rate: {sample_rate}")
            
            # Start the stream without using context manager
            stream = sd.play(data, sample_rate)
            
            with self.playback_lock:
                self.current_stream = stream
            
            # Wait until playback is finished
            sd.wait()
            logger.info("Audio playback completed")
        
        except Exception as e:
            logger.error(f"Error playing audio file: {str(e)}")

    def stop(self) -> None:
        """Stop the current playback."""
        with self.playback_lock:
            self.stop_requested = True
            if self.current_stream is not None:
                try:
                    # For sounddevice, use sd.stop() instead of stream.stop()
                    sd.stop()
                    logger.info("Audio playback stopped")
                except Exception as e:
                    logger.debug(f"Error stopping stream: {str(e)}")
                self.current_stream = None

    def is_audio_device_available(self) -> bool:
        """Check if an audio device is available for playback."""
        try:
            devices = sd.query_devices()
            return len(devices) > 0
        except Exception as e:
            logger.error(f"Error checking audio devices: {str(e)}")
            return False


# Singleton instance
_audio_player_instance = None


def get_audio_player() -> AudioPlayer:
    """
    Get the singleton audio player instance.
    
    Returns:
    AudioPlayer instance
    """
    global _audio_player_instance
    
    if _audio_player_instance is None:
        _audio_player_instance = AudioPlayer()
        
    return _audio_player_instance


def play_audio(file_path: Union[str, Path], blocking: bool = False) -> bool:
    """
    Play an audio file using the singleton audio player.
    
    Parameters:
    file_path: Path to the audio file
    blocking: If True, block until playback is complete
    
    Returns:
    True if playback started successfully, False otherwise
    """
    player = get_audio_player()
    
    # Debug information
    logger.info(f"play_audio called for file: {file_path}, blocking={blocking}")
    
    # Make sure file exists
    if not os.path.exists(file_path):
        logger.error(f"Audio file does not exist: {file_path}")
        return False
    
    # Try direct playback with soundfile/sounddevice
    try:
        # Load audio file directly
        logger.info(f"Loading audio file with soundfile: {file_path}")
        data, samplerate = sf.read(file_path)
        logger.info(f"File loaded: sample rate={samplerate}, shape={data.shape}")
        
        # Play the audio
        logger.info("Starting playback directly with sounddevice")
        sd.play(data, samplerate)
        
        if blocking:
            logger.info("Blocking until playback completes")
            sd.wait()
            logger.info("Playback completed")
        else:
            logger.info("Non-blocking playback initiated")
        
        return True
    except Exception as e:
        logger.error(f"Error playing audio directly: {str(e)}", exc_info=True)
        
        # Fall back to player implementation
        logger.info("Falling back to AudioPlayer.play method")
        return player.play(file_path, blocking)
