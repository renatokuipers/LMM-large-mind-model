# tts_module.py

from gradio_client import Client
from typing import Literal, Tuple
from dataclasses import dataclass
from typing import Dict


@dataclass
class TTSResult:
    audio_filepath: str
    phonemes: str

class TTSClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7860"):
        self.client = Client(base_url)

    def generate_speech(
        self,
        text: str,
        voice: str = "af"
    ) -> Dict[str, str]:
        """
        Generate speech audio from text.

        Parameters:
        - text: Text to synthesize.
        - voice: Voice selection from available options.

        Returns:
        {
            "audio_filepath": "<path_to_generated_audio>",
            "phonemes": "<phonetic transcription>"
        }
        """
        result = self.client.predict(
            text=text,
            voice=voice,
            api_name="/process_input"
        )
        audio_filepath, phonemes = result
        return {
            "audio_filepath": audio_filepath,
            "phonemes": phonemes
        }

    def available_voices(self) -> list:
        """List of available voices."""
        return [
            'af', 'af_bella', 'af_nicole', 'af_sarah', 'af_sky',
            'am_adam', 'am_michael', 'bf_emma', 'bf_isabella',
            'bm_george', 'bm_lewis'
        ]

# Usage Example
if __name__ == "__main__":
    tts_client = TTSClient()

    speech_result = tts_client.generate_speech(
        text="Look at the ball! Ball! It's a red ball!",
        voice="af_bella"
    )

    print("Generated audio file path:", speech_result["audio_filepath"])
    print("Phonemes transcription:", speech_result["phonemes"])
