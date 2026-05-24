"""
src/ablation_class_utterance.py — Class+Utterance controlled baseline.

Class-scoped FAISS indexes (same paths as runtime_rag.py), per-utterance
retrieval, ground-truth intent from t["intent"] — NO segmenter, NO classifier.
ONE Groq call per transcript.

This is the controlled ablation cell for Class+Utterance: it isolates the
index-scope and retrieval-granularity variables without the segmentation /
classification noise present in the full pipeline (data/results.json).

Prerequisites:
    data/indexes/ and data/maps/ must be populated (run build_indexes.py first).

Usage:
    C:\\v311\\Scripts\\python -m src.ablation_class_utterance
"""
import json
import time
from pathlib import Path

import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

from config import (
    GROQ_API_KEY, GROQ_MODEL,
    EMBEDDING_MODEL,
    INDEXES_DIR, MAPS_DIR,
    SIMILARITY_THRESHOLD, TOP_K,
)

ABLATION_DIR    = Path("data/ablation")
RESULTS_PATH    = ABLATION_DIR / "class_utterance_results.json"
TRANSCRIPTS_DIR = Path("data/transcripts")

GROQ_DELAY = 1.0

# Maps transcript intent names (short form) to actual BANKING77 index filenames.
# Mirrors the map in ablation_whole_call.py.
INTENT_MAP = {
    "balance_not_updated":   "balance_not_updated_after_bank_transfer",
    "disputed_charge":       "reverted_card_payment",
    "card_blocked":          "compromised_card",
    "failed_card_payment":   "declined_card_payment",
    "transfer_not_received": "transfer_not_received_by_recipient",
}

_index_cache: dict = {}


# ------------------------------------------------------------------
# Index loading — class-scoped, same paths as runtime_rag.py
# ------------------------------------------------------------------

def _load_index(intent: str):
    """Returns (index, policy_map) or (None, None) if index file not found."""
    resolved   = INTENT_MAP.get(intent, intent)
    if resolved in _index_cache:
        return _index_cache[resolved]
    index_path = Path(INDEXES_DIR) / f"{resolved}.faiss"
    map_path   = Path(MAPS_DIR)    / f"{resolved}.json"
    if not index_path.exists() or not map_path.exists():
        return None, None
    index      = faiss.read_index(str(index_path))
    policy_map = json.loads(map_path.read_text(encoding="utf-8"))
    _index_cache[resolved] = (index, policy_map)
    return index, policy_map


# ------------------------------------------------------------------
# Per-utterance retrieval against the class-scoped index
# ------------------------------------------------------------------

def _retrieve_utterance(utterance: str, embedder, index, policy_map) -> list[dict]:
    vec = embedder.encode([utterance], normalize_embeddings=True)
    vec = np.array(vec, dtype=np.float32)
    k_query = min(TOP_K, index.ntotal)
    scores, indices = index.search(vec, k_query)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if float(score) < SIMILARITY_THRESHOLD:
            continue
        results.append({
            "rule":  policy_map[idx]["rule"],
            "score": round(float(score), 4),
        })
    return results


# ------------------------------------------------------------------
# Single-transcript evaluation
# ------------------------------------------------------------------

def _evaluate_transcript(
    agent_turns: list[str],
    intent: str,
    index,
    policy_map,
    embedder,
    client,
) -> dict:
    # Per-utterance retrieval; deduplicate by rule text, keep highest score
    all_rules: dict[str, float] = {}
    for turn in agent_turns:
        for r in _retrieve_utterance(turn, embedder, index, policy_map):
            all_rules[r["rule"]] = max(all_rules.get(r["rule"], 0.0), r["score"])

    if all_rules:
        rules_text = "\n".join(f"- {rule}" for rule in all_rules)
        confidence = "High" if max(all_rules.values()) >= SIMILARITY_THRESHOLD else "Low"
    else:
        rules_text = "(no relevant policies retrieved above threshold)"
        confidence = "Low"

    turns_text = "\n".join(f'[{i+1}] "{turn}"' for i, turn in enumerate(agent_turns))

    # Identical prompt to runtime_rag.py
    prompt = (
        f"You are a banking call center QA evaluator.\n"
        f"Issue class: {intent}\n\n"
        f"AGENT TURNS (full conversation):\n{turns_text}\n\n"
        f"RELEVANT POLICIES:\n{rules_text}\n\n"
        f"Evaluate the full conversation. Did the agent violate any policy?\n"
        f"Only flag a violation if the agent clearly and directly broke a rule "
        f"and did NOT correct themselves later in the conversation.\n\n"
        f"Reply ONLY in this JSON format:\n"
        f'{{"verdict": "violation" or "ok", '
        f'"violations": [{{"turn": 1, "violated_policy": "...", "evidence": "...", "reason": "..."}}], '
        f'"overall_summary": "one sentence about the agent\'s overall performance"}}'
    )

    try:
        time.sleep(GROQ_DELAY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
    except Exception as exc:
        print(f"  [groq error] {exc} -- skipping")
        parsed = {"verdict": "error", "violations": [], "overall_summary": ""}

    return {
        "verdict":         parsed.get("verdict", "error"),
        "violations":      parsed.get("violations", []),
        "overall_summary": parsed.get("overall_summary", ""),
        "confidence":      confidence,
        "rules_retrieved": len(all_rules),
    }


def _policy_compliance(agent_turns: list[str], violations: list[dict]) -> float:
    total = len(agent_turns)
    if total == 0:
        return 0.0
    return round((total - len(violations)) / total * 100, 1)


# ------------------------------------------------------------------
# Batch runner
# ------------------------------------------------------------------

def run_eval(embedder, client) -> list[dict]:
    if RESULTS_PATH.exists():
        print(f"[eval] Cache hit — loading {RESULTS_PATH.name}")
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.json"))
    if not transcript_files:
        raise FileNotFoundError(f"No transcripts found in {TRANSCRIPTS_DIR}")

    n = len(transcript_files)
    print(f"[eval] Evaluating {n} transcripts (Class+Utterance controlled)...\n")

    results = []
    for i, fpath in enumerate(transcript_files, 1):
        t           = json.loads(fpath.read_text(encoding="utf-8"))
        agent_turns = [u["text"] for u in t["transcript"] if u["speaker"] == "agent"]

        print(f"  [{i:02d}/{n}] {t['call_id']}", end=" ", flush=True)

        index, policy_map = _load_index(t["intent"])
        if index is None:
            print(f"  [SKIP] no index found for intent '{t['intent']}'")
            continue

        eval_result = _evaluate_transcript(
            agent_turns, t["intent"], index, policy_map, embedder, client
        )
        compliance = _policy_compliance(agent_turns, eval_result["violations"])
        print(
            f"-> {eval_result['verdict'].upper()}"
            f"  violations={len(eval_result['violations'])}"
            f"  compliance={compliance}%"
        )

        results.append({
            "call_id":            t["call_id"],
            "intent":             t["intent"],
            "quality_level":      t["quality_level"],
            "planted_violations": t["planted_violations"],
            "verdict":            eval_result["verdict"],
            "violations":         eval_result["violations"],
            "overall_summary":    eval_result["overall_summary"],
            "confidence":         eval_result["confidence"],
            "rules_retrieved":    eval_result["rules_retrieved"],
            "policy_compliance":  compliance,
        })

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[eval] Saved to {RESULTS_PATH.resolve()}")
    return results


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    DIVIDER = "=" * 60
    print(DIVIDER)
    print("  Ablation Study — Class+Utterance Controlled Baseline")
    print(DIVIDER)

    print("\n[1/2] Loading embedding model and Groq client...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    client   = Groq(api_key=GROQ_API_KEY)
    print(f"      Embedding model : {EMBEDDING_MODEL}")
    print(f"      Similarity thresh: {SIMILARITY_THRESHOLD}  |  TOP_K: {TOP_K}")

    print("\n[2/2] Running Class+Utterance controlled evaluation...")
    run_eval(embedder, client)

    print()
    print("Done. Run ablation_single_faiss.py to see the full comparison table.")


if __name__ == "__main__":
    main()
