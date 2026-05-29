"""
src/generate_manuals.py — Generate policy manuals using real Banking77 examples.
"""
import json
import os
import random
from pathlib import Path

from openai import OpenAI
from datasets import load_dataset

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MANUALS_DIR

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MANUALS_PATH = Path(MANUALS_DIR)
ID2FINE_PATH = Path("model/id2fine.json")
SAMPLE_SIZE = 10
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


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
    examples_section = ""
    if examples:
        examples_text = "\n".join(f"- {ex}" for ex in examples)
        examples_section = f"Real customer queries for this issue:\n{examples_text}\n\n"

    return (
        f"You are a banking call center compliance expert.\n\n"
        f"Write a policy manual for agents handling: {fine_label}\n\n"
        f"{examples_section}"
        f"Write between 4 and 7 rules depending on the complexity of this issue.\n\n"
        f"Each rule must:\n"
        f"- Describe a SPECIFIC action the agent must take for THIS issue type\n"
        f"- Be measurable — someone can clearly tell if the agent did it or not\n"
        f"- Be unique to {fine_label} — not a generic rule that applies to all calls\n\n"
        f"DO NOT write rules about:\n"
        f"- Generic empathy or acknowledgment ('acknowledge the customer')\n"
        f"- Generic documentation ('document in account notes')\n"
        f"- Generic escalation ('escalate to supervisor if needed')\n"
        f"- Generic communication style ('communicate clearly')\n"
        f"- Customer consent or agreement ('obtain customer agreement')\n\n"
        f"GOOD rule example for cancel_transfer:\n"
        f"'- Rule 1: Initiate Recall Request: If transfer is still pending, "
        f"initiate a Faster Payment recall immediately and give customer a reference number.'\n\n"
        f"BAD rule example: '- Rule 1: Acknowledge and Empathize: Acknowledge the customer.'\n\n"
        f"Output ONLY the rules. No intro sentence. No conclusion sentence.\n"
        f"Format: each rule on its own line starting with '- Rule N: Rule Name: description'"
    )


def generate_manuals() -> None:
    print("Loading Banking77 dataset...")
    banking77 = build_banking77_lookup()
    norm_lookup = {_normalize(k): v for k, v in banking77.items()}

    fine_labels = list(
        json.loads(ID2FINE_PATH.read_text(encoding="utf-8")).values()
    )
    MANUALS_PATH.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
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
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
        )

        manual = response.choices[0].message.content.strip()
        safe_name = fine_label.replace("?", "")
        (MANUALS_PATH / f"{safe_name}.txt").write_text(manual, encoding="utf-8")

        print(f"[{i:02d}/{total}] OK  {fine_label}")

    print(f"\nDone. {total} manuals saved to {MANUALS_PATH.resolve()}")


if __name__ == "__main__":
    generate_manuals()
