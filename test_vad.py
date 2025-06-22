import torch
from silero_vad import get_speech_timestamps, read_audio, load_silero_vad, collect_chunks, save_audio
import silero_vad.utils_vad as utils_vad
from pathlib import Path

# --- Monkey-patch the OnnxWrapper __init__ to handle PosixPath bug ---
original_init = utils_vad.OnnxWrapper.__init__
def patched_init(self, path, *args, **kwargs):
    if isinstance(path, Path):
        path = str(path)  # Fix for 'in' operator bug
    original_init(self, path, *args, **kwargs)
utils_vad.OnnxWrapper.__init__ = patched_init

# --- Load model ---
model = load_silero_vad("cpu")

# --- Load audio file (must be 16kHz mono WAV) ---
audio = read_audio("test.wav", sampling_rate=16000)

# --- Run VAD ---
segments = get_speech_timestamps(audio, model, sampling_rate=16000)

# --- Print results ---
print("✅ Detected speech segments:")
for seg in segments:
    print(seg)

# --- Collect speech chunks and save ---
speech = collect_chunks(segments, audio)
save_audio("speech_only.wav", speech, sampling_rate=16000)
print("✅ Saved speech_only.wav")
