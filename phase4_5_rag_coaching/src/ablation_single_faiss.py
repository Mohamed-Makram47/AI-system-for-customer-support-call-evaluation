"""
src/ablation_single_faiss.py — Experiment 1: class-scoped FAISS vs single
monolithic FAISS index.
Only the retrieval scope changes; embedding model, similarity threshold, TOP_K,
Groq model, and evaluation prompt are identical to runtime_rag.py.
Usage:
    C:\\v311\\Scripts\\python src/ablation_single_faiss.py
"""
import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

from config import (
    GROQ_API_KEY, GROQ_MODEL,
    EMBEDDING_MODEL,
    MANUALS_DIR,
    SIMILARITY_THRESHOLD, TOP_K,
)

ABLATION_DIR        = Path("data/ablation")
SINGLE_INDEX_PATH   = ABLATION_DIR / "single_index.faiss"
SINGLE_MAP_PATH     = ABLATION_DIR / "single_map.json"
SINGLE_RESULTS_PATH = ABLATION_DIR / "single_faiss_results.json"
TRANSCRIPTS_DIR     = Path("data/transcripts")
CLASS_RESULTS_PATH  = Path("data/results.json")

GROQ_DELAY = 1.0


def _parse_rules(text: str) -> list[str]:
    rules = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-"):
            rule = re.sub(r"^-+\s*", "", stripped).strip()
            if rule:
                rules.append(rule)
    return rules


def build_single_index(embedder: SentenceTransformer) -> tuple:
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    if SINGLE_INDEX_PATH.exists() and SINGLE_MAP_PATH.exists():
        print("[single_index] Cache hit — loading existing index.")
        index      = faiss.read_index(str(SINGLE_INDEX_PATH))
        policy_map = json.loads(SINGLE_MAP_PATH.read_text(encoding="utf-8"))
        return index, policy_map

    manual_files = sorted(Path(MANUALS_DIR).glob("*.txt"))
    if not manual_files:
        raise FileNotFoundError(f"No .txt manuals found in {MANUALS_DIR}")

    print(f"[single_index] Reading {len(manual_files)} manuals...")
    all_rules: list[str]   = []
    policy_map: list[dict] = []
    for path in manual_files:
        source_class = path.stem
        rules = _parse_rules(path.read_text(encoding="utf-8"))
        for rule in rules:
            all_rules.append(rule)
            policy_map.append({"text": rule, "source_class": source_class})

    print(f"[single_index] {len(all_rules)} rules from {len(manual_files)} classes — embedding...")
    embeddings = embedder.encode(all_rules, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(SINGLE_INDEX_PATH))
    SINGLE_MAP_PATH.write_text(json.dumps(policy_map, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[single_index] Saved — {index.ntotal} vectors in index.")
    return index, policy_map


def _retrieve_single(utterance, embedder, index, policy_map):
    vec     = embedder.encode([utterance], normalize_embeddings=True)
    vec     = np.array(vec, dtype=np.float32)
    k_query = min(TOP_K, index.ntotal)
    scores, indices = index.search(vec, k_query)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if float(score) < SIMILARITY_THRESHOLD:
            continue
        entry = policy_map[idx]
        results.append({
            "rule":         entry["text"],
            "source_class": entry["source_class"],
            "score":        round(float(score), 4),
        })
    return results


def _evaluate_call_single(agent_turns, intent, embedder, index, policy_map, client):
    all_rules: dict[str, float] = {}
    for turn in agent_turns:
        for r in _retrieve_single(turn, embedder, index, policy_map):
            all_rules[r["rule"]] = max(all_rules.get(r["rule"], 0.0), r["score"])

    if all_rules:
        rules_text = "\n".join(f"- {rule}" for rule in all_rules)
        confidence = "High" if max(all_rules.values()) >= SIMILARITY_THRESHOLD else "Low"
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
        "rules_retrieved": len(all_rules),
    }


def _policy_compliance(agent_turns, violations):
    total = len(agent_turns)
    if total == 0:
        return 0.0
    return round((total - len(violations)) / total * 100, 1)


def run_single_faiss_eval(embedder, index, policy_map, client):
    if SINGLE_RESULTS_PATH.exists():
        print(f"[eval] Cache hit — loading {SINGLE_RESULTS_PATH.name}")
        return json.loads(SINGLE_RESULTS_PATH.read_text(encoding="utf-8"))

    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.json"))
    if not transcript_files:
        raise FileNotFoundError(f"No transcripts found in {TRANSCRIPTS_DIR}")

    n = len(transcript_files)
    print(f"[eval] Evaluating {n} transcripts with single FAISS index...\n")

    results = []
    for i, fpath in enumerate(transcript_files, 1):
        t = json.loads(fpath.read_text(encoding="utf-8"))
        agent_turns = [u["text"] for u in t["transcript"] if u["speaker"] == "agent"]

        print(f"  [{i:02d}/{n}] {t['call_id']}", end=" ", flush=True)
        eval_result = _evaluate_call_single(
            agent_turns, t["intent"], embedder, index, policy_map, client
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
    SINGLE_RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[eval] Saved to {SINGLE_RESULTS_PATH.resolve()}")
    return results


def _load_transcript_metadata():
    planted_map, quality_map = {}, {}
    for fpath in TRANSCRIPTS_DIR.glob("*.json"):
        t = json.loads(fpath.read_text(encoding="utf-8"))
        planted_map[t["call_id"]] = t["planted_violations"]
        quality_map[t["call_id"]] = t["quality_level"]
    return planted_map, quality_map


def _avg(values):
    if not values:
        return "N/A"
    return f"{round(sum(values) / len(values), 1)}%"


def compare_results(class_results, single_results):
    planted_map, quality_map = _load_transcript_metadata()

    cs_violations = sum(len(r["violations"]) for r in class_results)
    cs_fp = sum(
        1 for r in class_results
        if r["verdict"] == "violation" and len(planted_map.get(r["call_id"], [])) == 0
    )
    cs_compliance = [r["score"]["policy_compliance"] for r in class_results if "score" in r]
    cs_good = [
        r["score"]["policy_compliance"]
        for r in class_results
        if "score" in r and quality_map.get(r["call_id"]) == "good"
    ]
    cs_bad = [
        r["score"]["policy_compliance"]
        for r in class_results
        if "score" in r and quality_map.get(r["call_id"]) == "bad"
    ]

    sf_violations = sum(len(r["violations"]) for r in single_results)
    sf_fp = sum(
        1 for r in single_results
        if r["verdict"] == "violation" and len(planted_map.get(r["call_id"], [])) == 0
    )
    sf_compliance = [r["policy_compliance"] for r in single_results]
    sf_good = [
        r["policy_compliance"]
        for r in single_results
        if quality_map.get(r["call_id"]) == "good"
    ]
    sf_bad = [
        r["policy_compliance"]
        for r in single_results
        if quality_map.get(r["call_id"]) == "bad"
    ]

    W, COL = 28, 14
    divider = "-" * W + "+" + "-" * (COL + 2) + "+" + "-" * (COL + 1)
    top     = "=" * (W + COL * 2 + 6)

    print()
    print(top)
    print("  ABLATION STUDY -- Experiment 1: Index Scope")
    print(f"  Class-Scoped FAISS ({len(class_results)} calls)  vs  Single FAISS ({len(single_results)} calls)")
    print(top)
    print(f"{'Metric':<{W}}| {'Class-Scoped':>{COL}} | {'Single FAISS':>{COL}}")
    print(divider)

    rows = [
        ("Violations Caught",      cs_violations,       sf_violations),
        ("False Positives",        cs_fp,               sf_fp),
        ("Avg Compliance Score",   _avg(cs_compliance), _avg(sf_compliance)),
        ("Avg Score (good calls)", _avg(cs_good),       _avg(sf_good)),
        ("Avg Score (bad calls)",  _avg(cs_bad),        _avg(sf_bad)),
    ]
    for label, cs_val, sf_val in rows:
        print(f"{label:<{W}}| {str(cs_val):>{COL}} | {str(sf_val):>{COL}}")

    print(top)
    print()
    print("Notes:")
    print("  Metric = policy_compliance: (clean turns / total turns) x 100")
    print("  False Positive = violation verdict on a call with no planted violations")
    print(f"  Good calls n={len(cs_good)} (class-scoped) / {len(sf_good)} (single)")
    print(f"  Bad  calls n={len(cs_bad)} (class-scoped) / {len(sf_bad)} (single)")
    print()


def main():
    DIVIDER = "=" * 60
    print(DIVIDER)
    print("  Ablation Study -- Experiment 1: Single FAISS Index")
    print(DIVIDER)

    print("\n[1/3] Loading embedding model and Groq client...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    client   = Groq(api_key=GROQ_API_KEY)
    print(f"      Embedding model : {EMBEDDING_MODEL}")
    print(f"      Similarity thresh: {SIMILARITY_THRESHOLD}  |  TOP_K: {TOP_K}")

    print("\n[2/3] Building / loading single monolithic FAISS index...")
    index, policy_map = build_single_index(embedder)
    print(f"      Index size: {index.ntotal} vectors")

    print("\n[3/3] Running evaluation with single FAISS index...")
    single_results = run_single_faiss_eval(embedder, index, policy_map, client)

    print("\nLoading class-scoped results from data/results.json...")
    if not CLASS_RESULTS_PATH.exists():
        print("ERROR: data/results.json not found. Run main.py first.")
        return
    class_results = json.loads(CLASS_RESULTS_PATH.read_text(encoding="utf-8"))

    compare_results(class_results, single_results)


if __name__ == "__main__":
    main()
