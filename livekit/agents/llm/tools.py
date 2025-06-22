
import inspect
# livekit/agents/llm/tools.py

from livekit.agents.llm.openai import Tool




class FunctionTool(Tool):
    @classmethod
    def from_defaults(cls, fn):
        name = fn.__name__
        description = fn.__doc__ or "No description"
        sig = inspect.signature(fn)
        parameters = {
            k: {"type": "string", "description": str(v)} for k, v in sig.parameters.items()
        }
        return cls(
            name=name,
            description=description,
            fn=fn,
            parameters=parameters,
        )
