import os
import sys
import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
import time

def test_sounddevice_info():
    """Print out information about sound devices."""
    print("=== Sound Device Information ===")
    print(f"Default device: {sd.default.device}")
    print(f"Default samplerate: {sd.default.samplerate}")
    
    print("\nAvailable devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"[{i}] {device['name']} (in={device['max_input_channels']}, out={device['max_output_channels']})")
    
    return devices

def play_sine_wave():
    """Test playing a simple sine wave."""
    print("\n=== Playing Test Sine Wave ===")
    try:
        # Generate a sine wave
        samplerate = 44100
        duration = 2  # seconds
        frequency = 440  # Hz (A4)
        
        t = np.linspace(0, duration, int(samplerate * duration), False)
        tone = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        print("Playing sine wave...")
        sd.play(tone, samplerate)
        sd.wait()
        print("Sine wave playback completed")
        return True
    except Exception as e:
        print(f"Error playing sine wave: {str(e)}")
        return False

def test_file_playback(file_path=None):
    """Test playing an audio file."""
    print("\n=== Testing Audio File Playback ===")
    
    if file_path is None:
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
            file_path = wav_files[0]
            print(f"Using most recent WAV file: {file_path}")
        else:
            print("No WAV files found in the Kokoro cache directory.")
            return False
    
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: Audio file not found: {file_path}")
            return False
        
        # Get file info
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} bytes")
        
        # Read the file
        print(f"Loading audio file: {file_path}")
        data, samplerate = sf.read(file_path)
        print(f"File loaded successfully. Sample rate: {samplerate}, Shape: {data.shape}")
        
        # Play the audio
        print("Playing audio file...")
        sd.play(data, samplerate)
        sd.wait()
        print("Audio file playback completed")
        return True
    except Exception as e:
        print(f"Error playing audio file: {str(e)}")
        return False

def main():
    """Test audio system components."""
    print("Starting Audio System Test")
    print("-" * 50)
    
    # Test 1: Sound Device Information
    devices = test_sounddevice_info()
    if not devices:
        print("ERROR: No audio devices detected")
        return
    
    # Test 2: Sine Wave Playback
    sine_result = play_sine_wave()
    
    # Test 3: Audio File Playback
    if len(sys.argv) > 1:
        file_result = test_file_playback(sys.argv[1])
    else:
        file_result = test_file_playback()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Sound devices detected: {'Yes' if devices else 'No'}")
    print(f"Sine wave playback: {'Success' if sine_result else 'Failed'}")
    print(f"Audio file playback: {'Success' if file_result else 'Failed'}")

if __name__ == "__main__":
    main() 