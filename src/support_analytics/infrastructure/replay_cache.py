from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ReplayCache:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def build_key(
        self,
        *,
        namespace: str,
        model: str,
        seed: int,
        system_instruction: str,
        prompt: str,
        schema_name: str,
    ) -> str:
        fingerprint = sha256(
            json.dumps(
                {
                    "namespace": namespace,
                    "model": model,
                    "seed": seed,
                    "system_instruction": system_instruction,
                    "prompt": prompt,
                    "schema_name": schema_name,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return fingerprint

    def load(self, key: str, schema: type[T]) -> T | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        return schema.model_validate_json(path.read_text(encoding="utf-8"))

    def store(self, key: str, payload: BaseModel) -> None:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    def _path_for(self, key: str) -> Path:
        return self._root / key[:2] / f"{key}.json"

