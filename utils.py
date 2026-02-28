import json
from pathlib import Path
import random
from scheme import Chat

def pick_scenario(dict_scenarios):
    topic_key = random.choice(list(dict_scenarios.keys()))
    topic = dict_scenarios[topic_key]

    scenario_key = random.choice(list(topic.keys()))
    scenario = topic[scenario_key]

    return topic_key, scenario_key, scenario


def save_json(path: str, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False

def validate_chat(raw_json: str) -> Chat:
    data = json.loads(raw_json)
    return Chat.model_validate(data)  # pydantic v2