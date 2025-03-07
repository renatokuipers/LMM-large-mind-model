# tts_module.py
import os
import json
import tempfile
import shutil
import time
from typing import List, Optional, Dict, Any, Literal, Union
from pathlib import Path
from uuid import uuid4

import requests
from pydantic import BaseModel, Field, field_validator

LanguageCode = Literal['af', 'am', 'an', 'ar', 'as', 'az', 'ba', 'bg', 'bn', 'bpy', 'bs', 'ca', 'cmn', 'cs', 'cy', 'da', 'de', 'el', 'en-029', 'en-gb', 'en-gb-scotland', 'en-gb-x-gbclan', 'en-gb-x-gbcwmd', 'en-gb-x-rp', 'en-us', 'eo', 'es', 'es-419', 'et', 'eu', 'fa', 'fa-latn', 'fi', 'fr-be', 'fr-ch', 'fr-fr', 'ga', 'gd', 'gn', 'grc', 'gu', 'hak', 'hi', 'hr', 'ht', 'hu', 'hy', 'hyw', 'ia', 'id', 'is', 'it', 'ja', 'jbo', 'ka', 'kk', 'kl', 'kn', 'ko', 'kok', 'ku', 'ky', 'la', 'lfn', 'lt', 'lv', 'mi', 'mk', 'ml', 'mr', 'ms', 'mt', 'my', 'nb', 'nci', 'ne', 'nl', 'om', 'or', 'pa', 'pap', 'pl', 'pt', 'pt-br', 'py', 'quc', 'ro', 'ru', 'ru-lv', 'sd', 'shn', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'tn', 'tr', 'tt', 'ur', 'uz', 'vi', 'vi-vn-x-central', 'vi-vn-x-south', 'yue']
UnconditionalKey = Literal['speaker', 'emotion', 'vqscore_8', 'fmax', 'pitch_std', 'speaking_rate', 'dnsmos_ovrl', 'speaker_noised']
VoiceProfile = Literal['default', 'mother', 'father', 'child', 'elderly', 'robot']

OUTPUT_DIRECTORY = "generated"
PROFILE_FILENAME_FORMAT = "{profile}_voice.wav"
DEFAULT_FILENAME = "default_voice.wav"

class EmotionValues(BaseModel):
    happiness: float = Field(default=0.0, ge=0.0, le=1.0)
    sadness: float = Field(default=0.0, ge=0.0, le=1.0)
    disgust: float = Field(default=0.0, ge=0.0, le=1.0)
    fear: float = Field(default=0.0, ge=0.0, le=1.0)
    surprise: float = Field(default=0.0, ge=0.0, le=1.0)
    anger: float = Field(default=0.0, ge=0.0, le=1.0)
    other: float = Field(default=0.0, ge=0.0, le=1.0)
    neutral: float = Field(default=1.0, ge=0.0, le=1.0)

class VoiceParameters(BaseModel):
    vq_score: float = Field(default=0.78, ge=0.0, le=1.0)
    fmax_hz: int = Field(default=24000, ge=0)
    pitch_std: int = Field(default=45, ge=0)
    speaking_rate: int = Field(default=15, ge=0)
    dnsmos_overall: int = Field(default=4, ge=0)
    denoise_speaker: bool = Field(default=False)
    cfg_scale: float = Field(default=2.0, ge=0.0)
    min_p: float = Field(default=0.15, ge=0.0, le=1.0)
    seed: int = Field(default=420)
    randomize_seed: bool = Field(default=True)
    unconditional_keys: List[UnconditionalKey] = Field(default=["emotion"])

VOICE_PROFILES: Dict[VoiceProfile, Dict[str, Any]] = {
    "default": {},
    
    "mother": {
        "vq_score": 0.85,
        "fmax_hz": 28000,
        "pitch_std": 38,
        "speaking_rate": 14,
        "dnsmos_overall": 5,
        "cfg_scale": 2.5,
    },
    
    "father": {
        "vq_score": 0.82,
        "fmax_hz": 20000,
        "pitch_std": 35,
        "speaking_rate": 14,
        "dnsmos_overall": 5,
    },
    
    "child": {
        "vq_score": 0.75,
        "fmax_hz": 32000,
        "pitch_std": 60,
        "speaking_rate": 16,
        "dnsmos_overall": 3,
    },
    
    "elderly": {
        "vq_score": 0.72,
        "fmax_hz": 22000,
        "pitch_std": 30,
        "speaking_rate": 13,
        "dnsmos_overall": 3,
    },
    
    "robot": {
        "vq_score": 0.65,
        "fmax_hz": 20000,
        "pitch_std": 20,
        "speaking_rate": 16,
        "dnsmos_overall": 2,
        "cfg_scale": 3.0,
    }
}

class GenerateAudioRequest(BaseModel):
    model_type: str = Field(default="Zyphra/Zonos-v0.1-transformer")
    text: str
    language_code: LanguageCode = Field(default="en-us")
    speaker_audio: Optional[str] = None
    prefix_audio: Optional[str] = None
    emotions: EmotionValues = Field(default_factory=EmotionValues)
    voice_params: VoiceParameters = Field(default_factory=VoiceParameters)
    
    @field_validator('speaker_audio', 'prefix_audio')
    def validate_audio_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if v.startswith(('http://', 'https://')):
            return v
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Audio file not found: {v}")
        return str(path)

class ZonosClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7860"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        try:
            self.session.get(f"{self.base_url}", timeout=5).raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Could not connect to Zonos API at {self.base_url}")
    
    def _handle_file(self, file_path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not file_path:
            return None
        if file_path.startswith(('http://', 'https://')):
            return {"path": file_path}
        return {"path": file_path}
    
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
    
    def generate_audio(self, request: GenerateAudioRequest, save_to: Optional[str] = None) -> str:
        speaker_audio_param = self._handle_file(request.speaker_audio)
        prefix_audio_param = self._handle_file(request.prefix_audio)
        
        emotions = request.emotions
        voice = request.voice_params
        
        api_data = [
            request.model_type,
            request.text,
            request.language_code,
            speaker_audio_param,
            prefix_audio_param,
            emotions.happiness,
            emotions.sadness,
            emotions.disgust,
            emotions.fear,
            emotions.surprise,
            emotions.anger,
            emotions.other,
            emotions.neutral,
            voice.vq_score,
            voice.fmax_hz,
            voice.pitch_std,
            voice.speaking_rate,
            voice.dnsmos_overall,
            voice.denoise_speaker,
            voice.cfg_scale,
            voice.min_p,
            voice.seed,
            voice.randomize_seed,
            voice.unconditional_keys
        ]
        
        endpoint = "/gradio_api/call/generate_audio"
        
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
            raise ValueError(f"Invalid data content")
            
        audio_info = data_content[0] 
        seed = int(data_content[1])
        
        if isinstance(audio_info, dict) and 'path' in audio_info:
            audio_path = audio_info['path']
            
            if save_to:
                output_path = save_to
            else:
                temp_dir = tempfile.gettempdir()
                temp_filename = f"zonos_audio_{uuid4()}.wav"
                output_path = os.path.join(temp_dir, temp_filename)
            
            if os.path.exists(audio_path):
                self._wait_for_completion(audio_path)
            
            if os.path.exists(audio_path):
                shutil.copy2(audio_path, output_path)
                return output_path
            
            elif audio_path.startswith(('http://', 'https://')) or audio_info.get('url'):
                url = audio_path if audio_path.startswith(('http://', 'https://')) else audio_info.get('url')
                
                try:
                    response = self.session.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            
                    return output_path
                except:
                    pass
            
            return audio_path if os.path.exists(audio_path) else output_path
        
        raise ValueError("Invalid audio info format")
    
    def update_ui(self, model_type: str = "Zyphra/Zonos-v0.1-transformer") -> Dict[str, Any]:
        endpoint = "/gradio_api/call/update_ui"
        
        response = self.session.post(
            f"{self.base_url}{endpoint}",
            json={"data": [model_type]},
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
        
        if data_content and isinstance(data_content, list) and len(data_content) >= 19:
            return {
                "text": data_content[0],
                "language_code": data_content[1],
                "speaker_audio": data_content[2],
                "prefix_audio": data_content[3],
                "happiness": data_content[4],
                "sadness": data_content[5],
                "disgust": data_content[6],
                "fear": data_content[7],
                "surprise": data_content[8],
                "anger": data_content[9],
                "other": data_content[10],
                "neutral": data_content[11],
                "vq_score": data_content[12],
                "fmax_hz": data_content[13],
                "pitch_std": data_content[14],
                "speaking_rate": data_content[15],
                "dnsmos_overall": data_content[16],
                "denoise_speaker": data_content[17],
                "unconditional_keys": data_content[18]
            }
        
        return {}


def get_profile_output_path(profile: VoiceProfile) -> str:
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    if profile == "default":
        filename = DEFAULT_FILENAME
    else:
        filename = PROFILE_FILENAME_FORMAT.format(profile=profile)
    
    return os.path.join(OUTPUT_DIRECTORY, filename)


def apply_voice_profile(voice_params: Dict[str, Any], profile: VoiceProfile = "default") -> Dict[str, Any]:
    params = voice_params.copy()
    
    profile_settings = VOICE_PROFILES.get(profile, {})
    for key, value in profile_settings.items():
        params[key] = value
        
    return params


def text_to_speech(
    text: str, 
    language: LanguageCode = "en-us", 
    output_path: Optional[str] = None,
    emotions: Optional[Dict[str, float]] = None,
    voice_params: Optional[Dict[str, Any]] = None,
    voice_profile: VoiceProfile = "mother",
    speaker_audio_path: Optional[str] = None,
    model_type: str = "Zyphra/Zonos-v0.1-transformer",
    use_profile_path: bool = True
) -> str:
    client = ZonosClient()
    
    if output_path is None and use_profile_path:
        output_path = get_profile_output_path(voice_profile)
    
    request_dict = {
        "model_type": model_type,
        "text": text,
        "language_code": language
    }
    
    final_voice_params = {}
    
    final_voice_params = apply_voice_profile(final_voice_params, voice_profile)
    
    if voice_params:
        for key, value in voice_params.items():
            final_voice_params[key] = value
    
    if speaker_audio_path:
        request_dict["speaker_audio"] = speaker_audio_path
    
    if emotions:
        request_dict["emotions"] = EmotionValues(**emotions)
    
    if final_voice_params:
        request_dict["voice_params"] = VoiceParameters(**final_voice_params)
    
    request = GenerateAudioRequest(**request_dict)
    return client.generate_audio(request, save_to=output_path)


if __name__ == "__main__":
    audio_path = text_to_speech("It's time for dinner, sweetheart.")
    print(f"Mother voice saved to: {audio_path}")

    audio_path = text_to_speech(
        "I'm so proud of what you've accomplished!", 
        emotions={"happiness": 0.9, "neutral": 0.1}
    )
    print(f"Happy mother voice saved to: {audio_path}")

    for profile in ["father", "child", "elderly", "robot"]:
        audio_path = text_to_speech(
            f"This is a test of the {profile} voice profile.", 
            voice_profile=profile
        )
        print(f"{profile.title()} voice saved to: {audio_path}")