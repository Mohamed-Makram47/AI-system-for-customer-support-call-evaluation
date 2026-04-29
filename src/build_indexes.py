"""
src/build_indexes.py — Builds one FAISS IndexFlatIP per fine class.

For each manual in manuals/:
  - Parses rules (lines starting with '-')
  - Embeds each rule with all-MiniLM-L6-v2
  - Saves index  → data/indexes/{fine_label}.faiss
  - Saves map    → data/maps/{fine_label}.json
"""
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths (mirrors config.py)
MANUALS_DIR = Path("manuals")
INDEXES_DIR = Path("data/indexes")
MAPS_DIR    = Path("data/maps")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def slugify(label: str) -> str:
    return label.replace("/", "_").replace("?", "").replace(" ", "_")


def parse_rules(text: str) -> list[str]:
    """Extract lines that start with a dash (the 5 policy rules)."""
    rules = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-"):
            # Remove leading dash and clean up
            rule = re.sub(r"^-+\s*", "", stripped).strip()
            if rule:
                rules.append(rule)
    return rules


def build_indexes():
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    manual_files = sorted(MANUALS_DIR.glob("*.txt"))
    if not manual_files:
        raise FileNotFoundError(f"No .txt files found in {MANUALS_DIR}/")

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    total = len(manual_files)
    skipped, built = 0, 0

    for i, path in enumerate(manual_files, 1):
        fine_label = path.stem  # filename without .txt

        index_path = INDEXES_DIR / f"{fine_label}.faiss"
        map_path   = MAPS_DIR    / f"{fine_label}.json"

        if index_path.exists() and map_path.exists():
            print(f"[{i:02d}/{total}] SKIP  {fine_label}")
            skipped += 1
            continue

        text  = path.read_text(encoding="utf-8")
        rules = parse_rules(text)

        if not rules:
            print(f"[{i:02d}/{total}] WARN  {fine_label} — no rules parsed, skipping")
            skipped += 1
            continue

        # Embed rules → shape (n_rules, 384)
        embeddings = model.encode(rules, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Build IndexFlatIP (inner product = cosine on normalised vectors)
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Save index
        faiss.write_index(index, str(index_path))

        # Save metadata map: list of {rule, source}
        policy_map = [{"rule": rule, "source": fine_label} for rule in rules]
        map_path.write_text(json.dumps(policy_map, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[{i:02d}/{total}] OK    {fine_label}  ({len(rules)} rules)")
        built += 1

    print(f"\nDone. {built} built, {skipped} skipped.")
    print(f"Indexes -> {INDEXES_DIR.resolve()}")
    print(f"Maps    -> {MAPS_DIR.resolve()}")


if __name__ == "__main__":
    build_indexes()
