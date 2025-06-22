from typing import List, Optional
from .openai import ChatMessage
from pydantic import BaseModel
from typing import Any
from typing import Literal
from .openai import FunctionCall
from enum import Enum

class ChatRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"

class ChatMessage(BaseModel):
    role: ChatRole
    content: str
    id: Optional[str] = None
    type: str = "message"  # 👈 add this line



class ChatContent(BaseModel):
    content: Optional[str] = None

class ChatContext:
    def __init__(self, messages: Optional[List[ChatMessage]] = None):
        from livekit.agents.voice.generation import INSTRUCTIONS_MESSAGE_ID
        self.messages = messages or []
        

        if not any(msg.id == INSTRUCTIONS_MESSAGE_ID for msg in self.messages):
            self.messages.insert(0, ChatMessage(
                id=INSTRUCTIONS_MESSAGE_ID,
                role=ChatRole.system,
                content="You are a helpful travel assistant. Ask questions naturally.",
            ))

    def insert(self, message: ChatMessage):
        self.messages.insert(0, message)

    def add_message(self, *, role: ChatRole, content: str, id: Optional[str] = None, **kwargs):
        msg = ChatMessage(role=role, content=content, id=id, **kwargs)
        self.messages.append(msg)
        return msg


    def get_messages(self) -> List[ChatMessage]:
        return self.messages
    
    @property
    def items(self):
        return self.messages

    def clear(self):
        self.messages = []

    @classmethod
    def empty(cls):
        return cls(messages=[])
    
    def index_by_id(self, message_id: str) -> int:
        """Returns the index of the message with the given ID."""
        for i, item in enumerate(self.messages):
            if item.id == message_id:
                return i
        raise ValueError(f"Message ID {message_id} not found in chat context.")



class _ReadOnlyChatContext:
    """Stub for internal readonly chat context used by Agent."""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

class ChatItem(BaseModel):
    id: Optional[str] = None
    role: Optional[ChatRole] = None
    content: Optional[str] = None
    type: str = "message"



    




