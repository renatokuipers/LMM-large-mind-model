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
            
            # Check file extension
            if file_path.suffix.lower() == '.wav':
                self._play_wav_sync(file_path)
            else:
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
            # Check file extension
            if file_path.suffix.lower() == '.wav':
                self._play_wav_sync(file_path)
            else:
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
        
        Parameters:
        file_path: Path to the WAV file
        """
        try:
            with wave.open(str(file_path), 'rb') as wf:
                # Get WAV file properties
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                
                # Define callback for audio chunks
                def callback(outdata, frames, time, status):
                    if status:
                        logger.warning(f"Audio status: {status}")
                    if self.stop_requested:
                        raise sd.CallbackStop
                    data = wf.readframes(frames)
                    if len(data) == 0:
                        raise sd.CallbackStop
                    # Convert bytes to samples
                    import numpy as np
                    if sample_width == 2:
                        data = np.frombuffer(data, dtype=np.int16)
                    elif sample_width == 4:
                        data = np.frombuffer(data, dtype=np.int32)
                    else:
                        data = np.frombuffer(data, dtype=np.int8)
                    
                    # Reshape for channels
                    data = data.reshape(-1, n_channels)
                    if len(data) < len(outdata):
                        outdata[:len(data)] = data
                        outdata[len(data):] = 0
                        raise sd.CallbackStop
                    else:
                        outdata[:] = data
                
                # Start the stream
                with sd.OutputStream(
                    samplerate=sample_rate,
                    channels=n_channels,
                    callback=callback,
                    blocksize=1024
                ) as stream:
                    with self.playback_lock:
                        self.current_stream = stream
                    
                    # Wait until the stream is done or stopped
                    while stream.active and not self.stop_requested:
                        time.sleep(0.1)
        
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
            
            # Start the stream
            with sd.play(data, sample_rate) as stream:
                with self.playback_lock:
                    self.current_stream = stream
                
                # Wait until playback is finished
                while stream.active and not self.stop_requested:
                    time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error playing audio file: {str(e)}")

    def stop(self) -> None:
        """Stop the current playback."""
        with self.playback_lock:
            self.stop_requested = True
            if self.current_stream is not None:
                try:
                    self.current_stream.stop()
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
    return player.play(file_path, blocking)
