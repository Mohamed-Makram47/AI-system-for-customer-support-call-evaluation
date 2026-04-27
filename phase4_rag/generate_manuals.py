"""
generate_manuals.py — Generates one policy manual per fine class label
and saves each as manuals/{fine_label}.txt using the Groq API.
"""
import json
import os
import time
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

ID2FINE_PATH = "model/id2fine.json"
OUTPUT_DIR = Path("manuals")

PROMPT_TEMPLATE = (
    "You are a banking call center compliance expert. "
    "Write a policy manual for agents handling the issue: {fine_label}. "
    "Write exactly 5 clear rules the agent must follow. "
    "Each rule on its own line starting with a dash. "
    "Be specific and practical."
)

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


def slugify(label: str) -> str:
    return label.replace("/", "_").replace("?", "").replace(" ", "_")


def main():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in .env or environment.")

    with open(ID2FINE_PATH, "r") as f:
        id2fine: dict = json.load(f)

    labels = list(id2fine.values())
    OUTPUT_DIR.mkdir(exist_ok=True)

    client = Groq(api_key=api_key)

    total = len(labels)
    skipped, generated, failed = 0, 0, 0

    for i, label in enumerate(labels, 1):
        filename = OUTPUT_DIR / f"{slugify(label)}.txt"

        if filename.exists():
            print(f"[{i:02d}/{total}] SKIP  {label}")
            skipped += 1
            continue

        prompt = PROMPT_TEMPLATE.format(fine_label=label)
        print(f"[{i:02d}/{total}] GEN   {label} ...", end=" ", flush=True)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400,
            )
            content = response.choices[0].message.content.strip()
            filename.write_text(content, encoding="utf-8")
            print("OK")
            generated += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

        # Small pause to respect rate limits
        time.sleep(0.5)

    print(f"\nDone. {generated} generated, {skipped} skipped, {failed} failed.")
    print(f"Manuals saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
