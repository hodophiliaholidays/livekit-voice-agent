from abc import ABC, abstractmethod


class STT(ABC):
    @abstractmethod
    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        """Transcribe PCM audio to text."""
        pass
