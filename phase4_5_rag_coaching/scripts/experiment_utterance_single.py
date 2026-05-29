"""
Experiment: Utterance-level evaluation + Single global FAISS index.
Condition 4 of the 2x2 ablation matrix.
Saves results to data/experiments/utterance_single_results.json
"""
import json
import sys
from pathlib import Path

# Adjust system path to import modules from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EMBEDDING_MODEL
from scripts.transcripts import TRANSCRIPTS
from src.classifier import ClassifierPipeline
from scripts.retrievers import single_faiss
from scripts.evaluators import utterance_level
from scripts.classify_transcript import classify
from scripts.experiment_utils import print_summary, save_results, build_result, DIVIDER

# Define experiment directory and output file paths
EXPERIMENTS_DIR  = Path("data/experiments")
INDEX_PATH       = EXPERIMENTS_DIR / "single_index.faiss"
MAP_PATH         = EXPERIMENTS_DIR / "single_map.json"
RESULTS_PATH     = EXPERIMENTS_DIR / "utterance_single_results.json"

def main():
    if not INDEX_PATH.exists():
        print("[ERROR] Single index not found. Run scripts/build_single_index.py first.")
        sys.exit(1)

    print(DIVIDER)
    print("  Experiment 4: Utterance-Level + Single FAISS Index")
    print(DIVIDER)

    print("\nLoading single index...")
    index      = faiss.read_index(str(INDEX_PATH))
    policy_map = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    print(f"Index loaded: {index.ntotal} vectors")

    # Load components
    embedder   = SentenceTransformer(EMBEDDING_MODEL)
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    classifier = ClassifierPipeline()

    total_violations = 0
    all_results      = []

    # Run evaluations over transcripts
    for transcript in TRANSCRIPTS:
        call_id    = transcript["call_id"]
        utterances = transcript["utterances"]

        # Majority vote classification per transcript
        predicted_label = classify(utterances, classifier)

        # 1) Retrieve policies using single global FAISS index
        customer_text = " ".join(
            u["text"] for u in utterances if u["speaker"] == "customer"
        )
        query = f"{predicted_label} {customer_text}"
        policies = single_faiss.retrieve(query, embedder, index, policy_map)

        # 2) Evaluate using utterance-level
        result = utterance_level.evaluate(
            utterances, predicted_label, policies, client,
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )

        verdict_display = result.get("verdict", "ERROR").upper()
        print(f"{call_id[:50]:<50} {verdict_display:>10}")

        if result.get("verdict") == "violation":
            total_violations += 1

        all_results.append(build_result(call_id, predicted_label, result))

    print_summary(len(TRANSCRIPTS), total_violations)
    save_results(all_results, RESULTS_PATH)

if __name__ == "__main__":
    main()
