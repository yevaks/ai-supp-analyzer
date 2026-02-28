from dotenv import load_dotenv
from openai import OpenAI
import os
import logging
from scenarios import SCENARIOS
from prompts import QWEN_SYSTEM_PROMPT, build_user_prompt
from utils import save_json, pick_scenario, validate_chat
import datetime
import requests, base64
import json

load_dotenv()
logging.basicConfig(level=logging.INFO)
BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
API_KEY=os.getenv("QWEN_API_KEY")

def _generate_one():
    logging.info(f"Started {datetime.datetime.now()}")
    scenario = pick_scenario(SCENARIOS)
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }

    payload = {
        "model": "moonshotai/kimi-k2.5",
        "messages": [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role":"user","content":build_user_prompt(scenario)},
            {
                "role": "assistant",
                "content": "{",
                "partial": True
            },
        ],
        "max_tokens": 16384,
        "temperature": 1.00,
        "top_p": 1.00,
        "stream": False,
        "chat_template_kwargs": {"thinking": False},
    }
    logging.info(f"Send request {datetime.datetime.now()}")
    response = requests.post(BASE_URL, headers=headers, json=payload)
    logging.info(f"Got response {datetime.datetime.now()})")

    leading_text = "{"
    generated = response.json()['choices'][0]['message']['content']
    if not generated.lstrip().startswith("{"):
        full = leading_text + generated
        return full
    return generated


def generate_dataset(n: int = 10, n_max = 15, output_path: str = "dataset.json"):
    logging.info(f"Dataset generation started at {datetime.datetime.now()}")

    dataset = []
    attempts = 0
    while len(dataset) < n and attempts <= n_max:
        logging.info(f"Generating chat {attempts + 1}/{n}")
        attempts += 1
        try:
            raw_chat = _generate_one()
            print(raw_chat)
            parsed = validate_chat(raw_chat)  # має повернути Pydantic model
            dataset.append(parsed.model_dump())
        except Exception as e:
            logging.error(f"Attempt {attempts} failed: {e}")

    if len(dataset) < n:
        logging.warning(f"Only generated {len(dataset)} valid chats out of {n} after {attempts} attempts.")

    logging.info("Writing dataset to file...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset}, f, ensure_ascii=False, indent=2)

    logging.info(f"Dataset saved to {output_path} at {datetime.datetime.now()}")

generate_dataset()