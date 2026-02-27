import json
from pathlib import Path


def save_json(path: str, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))