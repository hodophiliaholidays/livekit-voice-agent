# livekit/agents/llm/tool_context.py

from typing import Optional, Dict, Any
from pydantic import BaseModel

from functools import wraps

def function_tool(fn):
    """Marks a function as a tool that can be called by the LLM."""
    fn._is_function_tool = True  # Mark function for later detection
    return fn

class ToolError(Exception):
    pass

class StopResponse(BaseModel):
    """Stub for tool stop response used in agent tool execution."""
    reason: str = "stop"
    metadata: dict[str, Any] = {}



class ToolContext(BaseModel):
    input: Optional[Dict[str, Any]] = None
    chat_history: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None

def is_function_tool(obj) -> bool:
    """Returns True if the given object is a valid function tool."""
    # Replace this logic with real checks if needed
    return hasattr(obj, "function") or callable(obj)

def is_raw_function_tool(obj) -> bool:
    """Returns True if the given object is a raw function tool."""
    return hasattr(obj, "__call__")
