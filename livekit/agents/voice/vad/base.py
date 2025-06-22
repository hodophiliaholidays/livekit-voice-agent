# livekit/agents/voice/vad/base.py

class VAD:
    def __init__(self, *args, **kwargs):
        pass

    def process(self, audio):
        raise NotImplementedError("This method should be overridden.")
