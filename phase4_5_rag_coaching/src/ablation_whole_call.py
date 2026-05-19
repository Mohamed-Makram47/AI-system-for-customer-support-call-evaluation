"""
src/ablation_whole_call.py — Experiment 2: Segmented evaluation (current system)
vs Whole-call evaluation (ablation baseline).

Current system: segments transcript, classifies each segment, evaluates each
segment against its own class policy.
This ablation: evaluates the entire transcript as one unit against the
ground-truth intent policy, using a single FAISS query and a single Groq call.

Usage:
    C:\\v311\\Scripts\\python -m src.ablation_whole_call
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

ABLATION_DIR           = Path("data/ablation")
WHOLE_RESULTS_PATH     = ABLATION_DIR / "whole_call_results.json"
TRANSCRIPTS_DIR        = Path("data/transcripts")
SEGMENTED_RESULTS_PATH = Path("data/results.json")

GROQ_DELAY = 1.0

# Maps transcript intent names (short form) to actual BANKING77 index names.
# Only needed for intents whose transcript name differs from the index filename.
INTENT_MAP = {
    "balance_not_updated":   "balance_not_updated_after_bank_transfer",
    "disputed_charge":       "reverted_card_payment",
    "card_blocked":          "compromised_card",
    "failed_card_payment":   "declined_card_payment",
    "transfer_not_received": "transfer_not_received_by_recipient",
}


# ------------------------------------------------------------------
# Index loading (class-scoped, same indexes as runtime_rag.py)
# ------------------------------------------------------------------

_index_cache: dict = {}

def _load_index(intent: str):
    resolved = INTENT_MAP.get(intent, intent)
    if resolved in _index_cache:
        return _index_cache[resolved]
    index_path = Path(INDEXES_DIR) / f"{resolved}.faiss"
    map_path   = Path(MAPS_DIR)    / f"{resolved}.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No FAISS index for intent: {intent} (resolved: {resolved})")
    if not map_path.exists():
        raise FileNotFoundError(f"No policy map for intent: {intent} (resolved: {resolved})")
    index      = faiss.read_index(str(index_path))
    policy_map = json.loads(map_path.read_text(encoding="utf-8"))
    _index_cache[resolved] = (index, policy_map)
    return index, policy_map


# ------------------------------------------------------------------
# Retrieval — one query with concatenated agent text
# ------------------------------------------------------------------

def _retrieve(text: str, embedder, index, policy_map) -> list[dict]:
    vec = embedder.encode([text], normalize_embeddings=True)
    vec = np.array(vec, dtype=np.float32)
    k_actual = min(TOP_K, index.ntotal)
    scores, indices = index.search(vec, k_actual)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if float(score) < SIMILARITY_THRESHOLD:
            continue
        entry = policy_map[idx]
        results.append({
            "rule":   entry["rule"],
            "source": entry["source"],
            "score":  round(float(score), 4),
        })
    return results


# ------------------------------------------------------------------
# Whole-call evaluation
# ------------------------------------------------------------------

def evaluate_whole_call(agent_turns: list[str], intent: str, embedder, client) -> dict:
    index, policy_map = _load_index(intent)

    # Single query: concatenate all agent turns
    concatenated = " ".join(agent_turns)
    retrieved    = _retrieve(concatenated, embedder, index, policy_map)

    if retrieved:
        rules_text = "\n".join(f"- {r['rule']}" for r in retrieved)
        confidence = "High" if max(r["score"] for r in retrieved) >= SIMILARITY_THRESHOLD else "Low"
    else:
        rules_text = "(no relevant policies retrieved above threshold)"
        confidence = "Low"

    turns_text = "\n".join(f'[{i+1}] "{turn}"' for i, turn in enumerate(agent_turns))

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
        "rules_retrieved": len(retrieved),
    }


# ------------------------------------------------------------------
# Batch evaluation
# ------------------------------------------------------------------

def _policy_compliance(agent_turns: list[str], violations: list[dict]) -> float:
    total = len(agent_turns)
    if total == 0:
        return 0.0
    return round((total - len(violations)) / total * 100, 1)


def run_whole_call_eval(embedder, client) -> list[dict]:
    if WHOLE_RESULTS_PATH.exists():
        print(f"[eval] Cache hit -- loading {WHOLE_RESULTS_PATH.name}")
        return json.loads(WHOLE_RESULTS_PATH.read_text(encoding="utf-8"))

    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.json"))
    if not transcript_files:
        raise FileNotFoundError(f"No transcripts found in {TRANSCRIPTS_DIR}")

    n = len(transcript_files)
    print(f"[eval] Evaluating {n} transcripts with whole-call approach...\n")

    results = []
    for i, fpath in enumerate(transcript_files, 1):
        t           = json.loads(fpath.read_text(encoding="utf-8"))
        agent_turns = [u["text"] for u in t["transcript"] if u["speaker"] == "agent"]

        print(f"  [{i:02d}/{n}] {t['call_id']}", end=" ", flush=True)
        eval_result = evaluate_whole_call(agent_turns, t["intent"], embedder, client)
        compliance  = _policy_compliance(agent_turns, eval_result["violations"])
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
            "agent_turn_count":   len(agent_turns),
            "verdict":            eval_result["verdict"],
            "violations":         eval_result["violations"],
            "overall_summary":    eval_result["overall_summary"],
            "confidence":         eval_result["confidence"],
            "rules_retrieved":    eval_result["rules_retrieved"],
            "policy_compliance":  compliance,
        })

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    WHOLE_RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[eval] Saved to {WHOLE_RESULTS_PATH.resolve()}")
    return results


# ------------------------------------------------------------------
# Comparison table
# ------------------------------------------------------------------

def _load_transcript_metadata():
    planted_map, quality_map, turn_count_map = {}, {}, {}
    for fpath in TRANSCRIPTS_DIR.glob("*.json"):
        t   = json.loads(fpath.read_text(encoding="utf-8"))
        cid = t["call_id"]
        planted_map[cid]    = t["planted_violations"]
        quality_map[cid]    = t["quality_level"]
        turn_count_map[cid] = sum(1 for u in t["transcript"] if u["speaker"] == "agent")
    return planted_map, quality_map, turn_count_map


def _avg(values: list) -> str:
    if not values:
        return "N/A"
    return f"{round(sum(values) / len(values), 1)}%"


def _count_early_violations(results: list[dict], turn_count_map: dict) -> int:
    """Violations flagged at turn <= floor(total_agent_turns / 2)."""
    count = 0
    for r in results:
        cid   = r["call_id"]
        total = turn_count_map.get(cid, 0)
        half  = max(1, total // 2)
        for v in r.get("violations", []):
            turn_num = v.get("turn")
            if isinstance(turn_num, int) and turn_num <= half:
                count += 1
    return count


def compare_results(seg_results: list[dict], wc_results: list[dict]) -> None:
    planted_map, quality_map, turn_count_map = _load_transcript_metadata()

    # ---- segmented metrics (use pre-computed score.policy_compliance) ----
    seg_violations = sum(len(r.get("violations", [])) for r in seg_results)
    seg_fp = sum(
        1 for r in seg_results
        if r.get("verdict") == "violation"
        and len(planted_map.get(r["call_id"], [])) == 0
    )
    seg_compliance = [r["score"]["policy_compliance"] for r in seg_results if "score" in r]
    seg_good = [
        r["score"]["policy_compliance"]
        for r in seg_results
        if "score" in r and quality_map.get(r["call_id"]) == "good"
    ]
    seg_bad = [
        r["score"]["policy_compliance"]
        for r in seg_results
        if "score" in r and quality_map.get(r["call_id"]) == "bad"
    ]
    seg_early = _count_early_violations(seg_results, turn_count_map)

    # ---- whole-call metrics ----
    wc_violations = sum(len(r.get("violations", [])) for r in wc_results)
    wc_fp = sum(
        1 for r in wc_results
        if r.get("verdict") == "violation"
        and len(planted_map.get(r["call_id"], [])) == 0
    )
    wc_compliance = [r["policy_compliance"] for r in wc_results]
    wc_good = [
        r["policy_compliance"]
        for r in wc_results
        if quality_map.get(r["call_id"]) == "good"
    ]
    wc_bad = [
        r["policy_compliance"]
        for r in wc_results
        if quality_map.get(r["call_id"]) == "bad"
    ]
    wc_early = _count_early_violations(wc_results, turn_count_map)

    W, COL = 28, 12
    divider = "-" * W + "+" + "-" * (COL + 2) + "+" + "-" * (COL + 1)
    top     = "=" * (W + COL * 2 + 6)

    print()
    print(top)
    print("  ABLATION STUDY -- Experiment 2: Evaluation Granularity")
    print(f"  Segmented ({len(seg_results)} calls)  vs  Whole-Call ({len(wc_results)} calls)")
    print(top)
    print(f"{'Metric':<{W}}| {'Segmented':>{COL}} | {'Whole-Call':>{COL}}")
    print(divider)

    rows = [
        ("Violations Caught",       seg_violations,       wc_violations),
        ("False Positives",         seg_fp,               wc_fp),
        ("Avg Compliance Score",    _avg(seg_compliance), _avg(wc_compliance)),
        ("Avg Score (good calls)",  _avg(seg_good),       _avg(wc_good)),
        ("Avg Score (bad calls)",   _avg(seg_bad),        _avg(wc_bad)),
        ("Early Violations Caught", seg_early,            wc_early),
    ]
    for label, sv, wv in rows:
        print(f"{label:<{W}}| {str(sv):>{COL}} | {str(wv):>{COL}}")

    print(top)
    print()
    print("Notes:")
    print("  Segmented compliance = pre-computed score.policy_compliance from data/results.json")
    print("  Whole-call compliance = (clean turns / total turns) x 100")
    print("  False Positive = violation verdict on a call with no planted violations")
    print("  Early Violation = violation flagged at turn <= floor(total_agent_turns / 2)")
    print("  Whole-call retrieval: single FAISS query on concatenated agent text")
    print(f"  Good calls n={len(seg_good)} (segmented) / {len(wc_good)} (whole-call)")
    print(f"  Bad  calls n={len(seg_bad)} (segmented) / {len(wc_bad)} (whole-call)")
    print()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    DIVIDER = "=" * 60
    print(DIVIDER)
    print("  Ablation Study -- Experiment 2: Whole-Call Evaluation")
    print(DIVIDER)

    print("\n[1/3] Loading embedding model and Groq client...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    client   = Groq(api_key=GROQ_API_KEY)
    print(f"      Embedding model : {EMBEDDING_MODEL}")
    print(f"      Similarity thresh: {SIMILARITY_THRESHOLD}  |  TOP_K: {TOP_K}")

    print("\n[2/3] Running whole-call evaluation...")
    wc_results = run_whole_call_eval(embedder, client)

    print("\n[3/3] Loading segmented results from data/results.json...")
    if not SEGMENTED_RESULTS_PATH.exists():
        print("ERROR: data/results.json not found. Run main.py first.")
        return
    seg_results = json.loads(SEGMENTED_RESULTS_PATH.read_text(encoding="utf-8"))

    compare_results(seg_results, wc_results)


if __name__ == "__main__":
    main()
