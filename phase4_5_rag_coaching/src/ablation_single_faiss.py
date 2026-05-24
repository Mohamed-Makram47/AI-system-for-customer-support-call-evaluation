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
SINGLE_UTT_PATH     = ABLATION_DIR / "single_utterance_results.json"
WHOLE_RESULTS_PATH  = ABLATION_DIR / "whole_call_results.json"
CLASS_UTT_PATH      = ABLATION_DIR / "class_utterance_results.json"
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
    # True call-level retrieval: concatenate ALL agent turns, ONE FAISS query.
    concatenated = " ".join(agent_turns)
    retrieved    = _retrieve_single(concatenated, embedder, index, policy_map)

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
    planted_map, quality_map, turn_count_map = {}, {}, {}
    for fpath in TRANSCRIPTS_DIR.glob("*.json"):
        t   = json.loads(fpath.read_text(encoding="utf-8"))
        cid = t["call_id"]
        planted_map[cid]    = t["planted_violations"]
        quality_map[cid]    = t["quality_level"]
        turn_count_map[cid] = sum(1 for u in t["transcript"] if u["speaker"] == "agent")
    return planted_map, quality_map, turn_count_map


def _avg(values):
    if not values:
        return "N/A"
    return f"{round(sum(values) / len(values), 1)}%"


def _count_early_violations(results: list, turn_count_map: dict) -> int:
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


def _classification_metrics(results: list, planted_map: dict) -> dict:
    """Call-level binary classification metrics.
    TP = flagged violation on a call with planted violations
    FP = flagged violation on a clean call
    FN = missed violation on a call with planted violations
    TN = correctly cleared a clean call
    """
    if not results:
        return {k: "N/A" for k in
                ["tp", "fp", "fn", "tn",
                 "precision", "recall", "f1",
                 "good_acc", "bad_acc"]}

    tp = fp = fn = tn = 0
    for r in results:
        has_violations = len(planted_map.get(r["call_id"], [])) > 0
        flagged        = r.get("verdict") == "violation"
        if   flagged     and     has_violations: tp += 1
        elif flagged     and not has_violations: fp += 1
        elif not flagged and     has_violations: fn += 1
        else:                                    tn += 1

    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 3) \
                if (precision + recall) > 0 else 0.0
    good_acc  = round(tn / (tn + fp), 3) if (tn + fp) > 0 else 0.0
    bad_acc   = recall  # TP / (TP + FN) — identical formula, clearer name

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "good_acc": good_acc, "bad_acc": bad_acc,
    }


def compare_results(sf_results, su_results, wc_results, cu_results, fs_results):
    """5-column ablation table — two sections.
    sf = Single+Call   su = Single+Utterance
    wc = Class+Call    cu = Class+Utterance (controlled, ground-truth intent)
    fs = Full System   (segmented pipeline, data/results.json)
    """
    planted_map, quality_map, turn_count_map = _load_transcript_metadata()

    def _get_comp(r, use_score_key):
        if use_score_key:
            s = r.get("score")
            return s["policy_compliance"] if isinstance(s, dict) else None
        return r.get("policy_compliance")

    def _metrics(results, use_score_key=False):
        violations = sum(len(r.get("violations", [])) for r in results)
        fp = sum(
            1 for r in results
            if r.get("verdict") == "violation"
            and len(planted_map.get(r["call_id"], [])) == 0
        )
        comps  = [(r["call_id"], _get_comp(r, use_score_key)) for r in results]
        all_c  = [c for _, c in comps if c is not None]
        good_c = [c for cid, c in comps if c is not None and quality_map.get(cid) == "good"]
        bad_c  = [c for cid, c in comps if c is not None and quality_map.get(cid) == "bad"]
        early  = _count_early_violations(results, turn_count_map)
        return violations, fp, all_c, good_c, bad_c, early

    sf = _metrics(sf_results)
    su = _metrics(su_results)
    wc = _metrics(wc_results)
    cu = _metrics(cu_results)                       # ground-truth intent, flat policy_compliance
    fs = _metrics(fs_results, use_score_key=True)   # full pipeline, nested score dict

    W, COL = 26, 12
    sep = "-" * W + ("+" + "-" * (COL + 2)) * 5
    top = "=" * (W + (COL + 3) * 5)

    # ------------------------------------------------------------------
    # TABLE 1 — retrieval & compliance metrics (unchanged)
    # ------------------------------------------------------------------
    print()
    print(top)
    print("  ABLATION STUDY — 2x2 Controlled + Full System")
    print("  Table 1: Retrieval & Compliance Metrics")
    print(top)
    print(
        f"{'Metric':<{W}}"
        f"| {'Single+Call':>{COL}} "
        f"| {'Single+Utt':>{COL}} "
        f"| {'Class+Call':>{COL}} "
        f"| {'Class+Utt':>{COL}} "
        f"| {'Full System':>{COL}}"
    )
    print(sep)

    rows1 = [
        ("Violations Caught",       sf[0], su[0], wc[0], cu[0], fs[0]),
        ("False Positives",         sf[1], su[1], wc[1], cu[1], fs[1]),
        ("Avg Compliance Score",    _avg(sf[2]), _avg(su[2]), _avg(wc[2]), _avg(cu[2]), _avg(fs[2])),
        ("Avg Score (good calls)",  _avg(sf[3]), _avg(su[3]), _avg(wc[3]), _avg(cu[3]), _avg(fs[3])),
        ("Avg Score (bad calls)",   _avg(sf[4]), _avg(su[4]), _avg(wc[4]), _avg(cu[4]), _avg(fs[4])),
        ("Early Violations Caught", sf[5], su[5], wc[5], cu[5], fs[5]),
    ]
    for label, v1, v2, v3, v4, v5 in rows1:
        print(f"{label:<{W}}| {str(v1):>{COL}} | {str(v2):>{COL}} | {str(v3):>{COL}} | {str(v4):>{COL}} | {str(v5):>{COL}}")

    print(top)
    print()
    print("Notes:")
    print("  Single+Call  : monolithic FAISS, whole-call query  -> single_faiss_results.json")
    print("  Single+Utt   : monolithic FAISS, per-utterance query -> single_utterance_results.json")
    print("  Class+Call   : class-scoped FAISS, whole-call query -> whole_call_results.json")
    print("  Class+Utt    : class-scoped FAISS, per-utterance query (ground truth) -> class_utterance_results.json")
    print("  Full System  : class-scoped FAISS, per-utterance query -> data/results.json")
    print("  Compliance   = (clean turns / total turns) x 100")
    print("  False Positive = violation verdict on a call with no planted violations")
    print("  Early Violation = flagged at turn <= floor(total_agent_turns / 2)")
    print("  Full System uses automatic topic segmentation + per-segment classification.")
    print("  All other columns use ground truth intent labels with no segmentation")
    print("  -- controlled ablation conditions.")
    n_good = len(sf[3]) or len(su[3]) or len(wc[3]) or len(cu[3]) or len(fs[3])
    n_bad  = len(sf[4]) or len(su[4]) or len(wc[4]) or len(cu[4]) or len(fs[4])
    print(f"  n(good calls) = {n_good}  |  n(bad calls) = {n_bad}")

    # ------------------------------------------------------------------
    # TABLE 2 — call-level classification metrics
    # ------------------------------------------------------------------
    sf_cm = _classification_metrics(sf_results, planted_map)
    su_cm = _classification_metrics(su_results, planted_map)
    wc_cm = _classification_metrics(wc_results, planted_map)
    cu_cm = _classification_metrics(cu_results, planted_map)
    fs_cm = _classification_metrics(fs_results, planted_map)

    def _f(v):
        """Format metric value: int -> str, float -> 3 dp, N/A -> N/A."""
        if v == "N/A":
            return "N/A"
        if isinstance(v, int):
            return str(v)
        return f"{v:.3f}"

    print()
    print(top)
    print("  Table 2: Call-Level Classification Metrics")
    print(top)
    print(
        f"{'Metric':<{W}}"
        f"| {'Single+Call':>{COL}} "
        f"| {'Single+Utt':>{COL}} "
        f"| {'Class+Call':>{COL}} "
        f"| {'Class+Utt':>{COL}} "
        f"| {'Full System':>{COL}}"
    )
    print(sep)

    rows2 = [
        ("TP",             sf_cm["tp"],        su_cm["tp"],        wc_cm["tp"],        cu_cm["tp"],        fs_cm["tp"]),
        ("FP",             sf_cm["fp"],        su_cm["fp"],        wc_cm["fp"],        cu_cm["fp"],        fs_cm["fp"]),
        ("FN",             sf_cm["fn"],        su_cm["fn"],        wc_cm["fn"],        cu_cm["fn"],        fs_cm["fn"]),
        ("TN",             sf_cm["tn"],        su_cm["tn"],        wc_cm["tn"],        cu_cm["tn"],        fs_cm["tn"]),
        ("Precision",      sf_cm["precision"], su_cm["precision"], wc_cm["precision"], cu_cm["precision"], fs_cm["precision"]),
        ("Recall",         sf_cm["recall"],    su_cm["recall"],    wc_cm["recall"],    cu_cm["recall"],    fs_cm["recall"]),
        ("F1 Score",       sf_cm["f1"],        su_cm["f1"],        wc_cm["f1"],        cu_cm["f1"],        fs_cm["f1"]),
        ("Good Call Acc.", sf_cm["good_acc"],  su_cm["good_acc"],  wc_cm["good_acc"],  cu_cm["good_acc"],  fs_cm["good_acc"]),
        ("Bad Call Acc.",  sf_cm["bad_acc"],   su_cm["bad_acc"],   wc_cm["bad_acc"],   cu_cm["bad_acc"],   fs_cm["bad_acc"]),
    ]
    for label, v1, v2, v3, v4, v5 in rows2:
        print(f"{label:<{W}}| {_f(v1):>{COL}} | {_f(v2):>{COL}} | {_f(v3):>{COL}} | {_f(v4):>{COL}} | {_f(v5):>{COL}}")

    print(top)

    # Summary — best per key metric across all 5 columns
    col_names = ["Single+Call", "Single+Utt", "Class+Call", "Class+Utt", "Full System"]
    cms       = [sf_cm, su_cm, wc_cm, cu_cm, fs_cm]

    def _best(key):
        valid = [(col_names[i], cm[key]) for i, cm in enumerate(cms)
                 if isinstance(cm[key], float)]
        if not valid:
            return "N/A", "N/A"
        name, val = max(valid, key=lambda x: x[1])
        return name, f"{val:.3f}"

    bf_name, bf_val = _best("f1")
    br_name, br_val = _best("recall")
    bp_name, bp_val = _best("precision")

    print()
    print(f"  Best F1       : {bf_name} ({bf_val})")
    print(f"  Best Recall   : {br_name} ({br_val})")
    print(f"  Best Precision: {bp_name} ({bp_val})")
    print()


def main():
    DIVIDER = "=" * 60
    print(DIVIDER)
    print("  Ablation Study -- Full 2x2 Controlled + Full System")
    print(DIVIDER)

    print("\n[1/3] Loading embedding model and Groq client...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    client   = Groq(api_key=GROQ_API_KEY)
    print(f"      Embedding model : {EMBEDDING_MODEL}")
    print(f"      Similarity thresh: {SIMILARITY_THRESHOLD}  |  TOP_K: {TOP_K}")

    print("\n[2/3] Building / loading single monolithic FAISS index...")
    index, policy_map = build_single_index(embedder)
    print(f"      Index size: {index.ntotal} vectors")

    print("\n[3/3] Running Single+Call evaluation...")
    sf_results = run_single_faiss_eval(embedder, index, policy_map, client)

    su_results, wc_results, cu_results, fs_results = [], [], [], []
    missing = []

    if SINGLE_UTT_PATH.exists():
        su_results = json.loads(SINGLE_UTT_PATH.read_text(encoding="utf-8"))
        print(f"\nLoaded Single+Utterance results ({len(su_results)} calls)")
    else:
        missing.append(f"  Single+Utt  : run ablation_single_utterance.py -> {SINGLE_UTT_PATH}")

    if WHOLE_RESULTS_PATH.exists():
        wc_results = json.loads(WHOLE_RESULTS_PATH.read_text(encoding="utf-8"))
        print(f"Loaded Class+Call results ({len(wc_results)} calls)")
    else:
        missing.append(f"  Class+Call  : run ablation_whole_call.py -> {WHOLE_RESULTS_PATH}")

    if CLASS_UTT_PATH.exists():
        cu_results = json.loads(CLASS_UTT_PATH.read_text(encoding="utf-8"))
        print(f"Loaded Class+Utterance (controlled) results ({len(cu_results)} calls)")
    else:
        missing.append(f"  Class+Utt   : run ablation_class_utterance.py -> {CLASS_UTT_PATH}")

    if CLASS_RESULTS_PATH.exists():
        fs_results = json.loads(CLASS_RESULTS_PATH.read_text(encoding="utf-8"))
        print(f"Loaded Full System results ({len(fs_results)} calls)")
    else:
        missing.append(f"  Full System : run main.py -> {CLASS_RESULTS_PATH}")

    if missing:
        print("\n[WARNING] Missing result files -- those columns will show N/A:")
        for m in missing:
            print(m)

    compare_results(sf_results, su_results, wc_results, cu_results, fs_results)


if __name__ == "__main__":
    main()
