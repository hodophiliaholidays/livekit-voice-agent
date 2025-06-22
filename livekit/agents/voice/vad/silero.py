import torch
from silero_vad import read_audio, get_speech_timestamps, load_silero_vad
from silero_vad import load_silero_vad as load_model
import types


class SileroVAD:
    def __init__(self, sampling_rate=16000):
        self.sampling_rate = sampling_rate
        self.model = load_model()


    def get_speech_segments(self, audio_tensor: torch.Tensor):
        return get_speech_timestamps(audio_tensor, self.model, sampling_rate=self.sampling_rate)
    
    def stream(self):
        # Dummy async generator for compatibility
        async def dummy_stream():
            if False:
                yield
        return dummy_stream()


