import os
import argparse
import json
import google.genai as genai
from dotenv import load_dotenv

from prompts import build_analysis_prompt
from utils import load_json, save_json


load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", default="dataset.json")
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")

    client = genai.Client(api_key=api_key)

    dataset = load_json(args.infile)["dataset"]

    results = []

    for dialog in dataset:
        dialog_text = "\n".join(
            f'{turn["role"]}: {turn["text"]}'
            for turn in dialog["turns"]
        )

        prompt = build_analysis_prompt(dialog_text)

        response = client.models.generate_content(
            model="gemini-3.0-flash",
            contents=prompt,
            generation_config={
                "temperature": 0,
                "top_p": 1
            }
        )

        raw = json.loads(response.text.strip())

        results.append({
            "dialog_id": dialog["chat_id"],
            **raw
        })

    save_json(args.out, results)
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()