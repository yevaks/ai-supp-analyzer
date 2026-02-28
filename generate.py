import os
import argparse
import json
import google.genai as genai
from dotenv import load_dotenv
from scenarios import SCENARIOS
from prompts import build_generation_prompt
from utils import save_json, pick_scenario

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out", default="dataset.json")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")

    client = genai.Client(api_key=api_key)

    prompt = build_generation_prompt(args.n)

    response = client.models.generate_content(
        model="gemini-3.0-flash",
        contents=prompt,
        generation_config={
            "temperature": 0,
            "top_p": 1
        }
    )

    text = response.text.strip()

    data = json.loads(text)

    save_json(args.out, data)
    print(f"Dataset saved to {args.out}")


if __name__ == "__main__":
    main()