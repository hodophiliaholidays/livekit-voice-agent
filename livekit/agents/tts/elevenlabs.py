import os
import requests

class ElevenLabsTTS:
    def __init__(self, voice_id: str):
        self.voice_id = voice_id
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        self.url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"

    def synthesize(self, text: str) -> bytes:
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
        }
        response = requests.post(self.url, headers=headers, json=data)
        response.raise_for_status()
        return response.content
