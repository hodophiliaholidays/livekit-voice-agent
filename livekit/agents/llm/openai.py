from __future__ import annotations
from typing import Optional, Dict, Any, Callable, List, Literal
from pydantic import BaseModel, ConfigDict
from openai import AsyncOpenAI


# ─── Exception Definitions ─────────────────────────────────────────────────────
class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass

class ToolError(Exception):
    def __init__(self, tool_name: str, message: str):
        super().__init__(f"Tool '{tool_name}' failed: {message}")
        self.tool_name = tool_name
        self.message = message

class RealtimeModelError(Exception):
    """Custom exception for realtime model errors."""
    pass

# ─── OpenAI Chat Client ────────────────────────────────────────────────────────
class OpenAILLM:
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def chat(self, message: str) -> str:
        res = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}]
        )
        return res.choices[0].message.content
    
    def on(self, event: str, handler):
        # Stub method to avoid errors when metrics hooks are attached
        pass

    def prewarm(self):
        # No-op method to satisfy LiveKit agent
        pass

LLM = OpenAILLM  # Aliased for internal references

# ─── Tool Abstraction ──────────────────────────────────────────────────────────
class Tool:
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        fn: Optional[Callable] = None,
        parameters: Optional[Dict[str, Any]] = None,
        required: Optional[List[str]] = None
    ):
        self.name = name
        self.description = description
        self.fn = fn
        self.parameters = parameters or {}
        self.required = required or []

    @classmethod
    def from_defaults(cls, fn: Callable, name: Optional[str] = None, description: Optional[str] = None):
        return cls(
            name=name or fn.__name__,
            description=description or fn.__doc__ or "No description provided.",
            fn=fn
        )

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required": self.required,
            "function": self.fn
        }

# ─── Pydantic-Based FunctionTool ───────────────────────────────────────────────
class FunctionTool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = {}
    required: List[str] = []
    fn: Optional[Callable] = None

    def to_tool(self) -> Tool:
        return Tool(
            name=self.name,
            description=self.description,
            fn=self.fn,
            parameters=self.parameters,
            required=self.required
        )

    @classmethod
    def from_defaults(cls, fn: Callable, name: Optional[str] = None, description: Optional[str] = None):
        return cls(
            name=name or fn.__name__,
            description=description or fn.__doc__ or "No description provided.",
            fn=fn
        )

# ─── Supporting Models ─────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

class FunctionCall(BaseModel):
    name: str
    arguments: str

class FunctionCallOutput(BaseModel):
    name: str
    result: Any

class RawFunctionTool(BaseModel):
    name: str
    description: Optional[str]
    parameters: Dict[str, Any]

class RealtimeModel:
    """Stub for RealtimeModel (not implemented)"""
    pass

# ─── Tool Finder ───────────────────────────────────────────────────────────────
def find_function_tools(tools: List[FunctionTool]) -> Dict[str, FunctionTool]:
    return {tool.name: tool for tool in tools}
