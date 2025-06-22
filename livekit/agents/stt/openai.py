import os
import asyncio
import aiohttp
import types
from livekit.agents.stt.base import STT
from livekit.agents.utils.audio import pcm_to_wav_bytes



class OpenAIWhisperSTT(STT):
    def __init__(self, model: str = "whisper-1", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key")
        
    def on(self, event_name, callback):
        # Dummy event handler (does nothing)
        pass

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        wav_bytes = pcm_to_wav_bytes(pcm, sample_rate)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        data = aiohttp.FormData()
        data.add_field("file", wav_bytes, filename="audio.wav", content_type="audio/wav")
        data.add_field("model", self.model)
        data.add_field("language", "en")

        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/audio/transcriptions", headers=headers, data=data) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"OpenAI STT failed ({resp.status}): {text}")
                result = await resp.json()
                return result["text"]

    from types import SimpleNamespace

    @property
    def capabilities(self):
        return types.SimpleNamespace(streaming=False)
