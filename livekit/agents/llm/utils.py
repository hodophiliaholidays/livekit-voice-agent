# livekit/agents/llm/utils.py

def get_stop_response(*args, **kwargs):
    # Dummy fallback
    return {"type": "stop", "reason": "not_implemented"}

class StopResponse:
    def __init__(self, reason: str):
        self.reason = reason

    def dict(self):
        return {"type": "stop", "reason": self.reason}
