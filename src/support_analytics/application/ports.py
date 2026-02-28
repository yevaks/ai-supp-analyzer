from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    def generate_structured(
        self,
        *,
        model: str,
        prompt: str,
        system_instruction: str,
        response_schema: type[T],
        seed: int,
    ) -> T: ...

