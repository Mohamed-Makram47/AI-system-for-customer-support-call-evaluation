"""
src/generate_manuals.py — Generate policy manuals using real Banking77 examples.
"""
import json
import os
import random
import time
from pathlib import Path

from groq import Groq
from datasets import load_dataset

from config import GROQ_API_KEY, GROQ_MODEL, MANUALS_DIR

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MANUALS_PATH = Path(MANUALS_DIR)
ID2FINE_PATH = Path("model/id2fine.json")
SAMPLE_SIZE = 10
DELAY = 0.5


def build_banking77_lookup() -> dict:
    ds = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True)
    label_names = ds.features["label"].names
    lookup = {name: [] for name in label_names}
    for example in ds:
        lookup[label_names[example["label"]]].append(example["text"])
    return lookup


def _normalize(name: str) -> str:
    return name.lower().rstrip("?")


def _make_prompt(fine_label: str, examples: list) -> str:
    if examples:
        examples_text = "\n".join(f"- {ex}" for ex in examples)
        return (
            f"You are a banking call center compliance expert.\n"
            f"Here are real customer queries for the issue type: {fine_label}\n\n"
            f"Real examples:\n{examples_text}\n\n"
            f"Based on these real queries, write a policy manual for agents handling this issue.\n"
            f"Write exactly 5 clear rules the agent must follow.\n"
            f"Each rule on its own line starting with a dash.\n"
            f"Be specific and practical."
        )
    return (
        f"You are a banking call center compliance expert.\n"
        f"Write a policy manual for agents handling the issue type: {fine_label}\n\n"
        f"Write exactly 5 clear rules the agent must follow.\n"
        f"Each rule on its own line starting with a dash.\n"
        f"Be specific and practical."
    )


def generate_manuals() -> None:
    print("Loading Banking77 dataset...")
    banking77 = build_banking77_lookup()
    norm_lookup = {_normalize(k): v for k, v in banking77.items()}

    fine_labels = list(
        json.loads(ID2FINE_PATH.read_text(encoding="utf-8")).values()
    )
    MANUALS_PATH.mkdir(parents=True, exist_ok=True)

    client = Groq(api_key=GROQ_API_KEY)
    total = len(fine_labels)

    for i, fine_label in enumerate(fine_labels, start=1):
        raw_examples = norm_lookup.get(_normalize(fine_label), [])
        examples = (
            random.sample(raw_examples, min(SAMPLE_SIZE, len(raw_examples)))
            if raw_examples
            else []
        )

        prompt = _make_prompt(fine_label, examples)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )

        manual = response.choices[0].message.content.strip()
        safe_name = fine_label.replace("?", "")
        (MANUALS_PATH / f"{safe_name}.txt").write_text(manual, encoding="utf-8")

        print(f"[{i:02d}/{total}] OK  {fine_label}")

        if i < total:
            time.sleep(DELAY)

    print(f"\nDone. {total} manuals saved to {MANUALS_PATH.resolve()}")


if __name__ == "__main__":
    generate_manuals()
