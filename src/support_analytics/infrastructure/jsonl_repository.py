from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JsonlRepository:
    def write_jsonl(self, path: Path, items: list[BaseModel]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(self._serialize_model(item), ensure_ascii=False) for item in items]
        path.write_text("\n".join(lines), encoding="utf-8")

    def read_jsonl(self, path: Path, schema: type[T]) -> list[T]:
        if not path.exists():
            return []
        items: list[T] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(schema.model_validate_json(line))
        return items

    def append_jsonl(self, path: Path, item: BaseModel) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(self._serialize_model(item), ensure_ascii=False)
        prefix = "\n" if path.exists() and path.stat().st_size > 0 else ""
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{prefix}{line}")

    def write_json(self, path: Path, item: BaseModel | dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._serialize_model(item) if isinstance(item, BaseModel) else item
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_json(self, path: Path, schema: type[T]) -> T | None:
        if not path.exists():
            return None
        return schema.model_validate_json(path.read_text(encoding="utf-8"))

    def _serialize_model(self, item: BaseModel) -> dict[str, object]:
        return {
            field_name: self._serialize_value(getattr(item, field_name))
            for field_name in item.__class__.model_fields
        }

    def _serialize_value(self, value: object) -> object:
        if isinstance(value, BaseModel):
            return self._serialize_model(value)
        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._serialize_value(item) for key, item in value.items()}
        if isinstance(value, Enum):
            return value.value
        return value
