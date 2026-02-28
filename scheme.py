from pydantic import BaseModel, Field
from typing import List


class Message(BaseModel):
    role: str
    text: str


class Chat(BaseModel):
    chat_id: str
    scenario: str
    messages: List[Message]