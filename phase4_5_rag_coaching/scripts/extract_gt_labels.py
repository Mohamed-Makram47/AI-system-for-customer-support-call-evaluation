import sys
from pathlib import Path

# Adjust system path to import modules from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.transcripts import TRANSCRIPTS


def main():
    # Columns: call_id (51 chars), intent (50 chars including leading space), quality (8 chars including leading space)
    col1_header = f"{'call_id':<51}"
    col2_header = f" {'intent':<49}"
    col3_header = f" {'quality':<7}"

    print(f"{col1_header}|{col2_header}|{col3_header}")
    print("-" * 51 + "|" + "-" * 50 + "|" + "-" * 8)

    intents = set()
    missing_intents = 0

    for transcript in TRANSCRIPTS:
        call_id = transcript.get("call_id", "")
        intent = transcript.get("intent")
        quality = transcript.get("quality_level", "")

        if intent is None:
            missing_intents += 1
            intent_str = ""
        else:
            intents.add(intent)
            intent_str = intent

        col1 = f"{call_id:<51}"
        col2 = f" {intent_str:<49}"
        col3 = f" {quality:<7}"
        print(f"{col1}|{col2}|{col3}")

    print()
    print(f"Total: {len(TRANSCRIPTS)} transcripts")
    print(f"Unique intents: {len(intents)}")
    print(f"Missing intent field: {missing_intents}")


if __name__ == "__main__":
    main()
