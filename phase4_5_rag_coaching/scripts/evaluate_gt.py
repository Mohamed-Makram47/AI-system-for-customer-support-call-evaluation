"""Ground truth evaluation script.
Computes TP, FP, FN, Precision, Recall, F1 for each experiment condition
by matching detected violations against planted violations using LLM.
"""

import json
import sys
from pathlib import Path

import groq

from groq import Groq

# ---------------------------------------------------------------------------
# Allow running from the project root  (python scripts/evaluate_gt.py ...)
# or from the scripts/ directory       (python evaluate_gt.py ...)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import GROQ_API_KEY  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TRANSCRIPTS_DIR = _PROJECT_ROOT / "data" / "transcripts"
EXPERIMENTS_DIR = _PROJECT_ROOT / "data" / "experiments"

# Quality levels that carry planted violations (skip good, noise, recovery)
EVAL_QUALITY_LEVELS = {"bad", "ambiguous", "incomplete"}
CLEAN_QUALITY_LEVELS = {"good", "noise", "recovery"}

DIVIDER = "=" * 70
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
client = Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_transcripts() -> dict:
    """Load all transcript JSON files and return a dict keyed by call_id.

    Only transcripts whose quality_level is in EVAL_QUALITY_LEVELS are kept.
    """
    transcripts: dict = {}
    for path in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        quality = data.get("quality_level", "")
        if quality in EVAL_QUALITY_LEVELS:
            transcripts[data["call_id"]] = data
    return transcripts


def load_clean_transcripts() -> dict:
    """Load transcripts with no planted violations (good/noise/recovery)."""
    transcripts: dict = {}
    for path in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        quality = data.get("quality_level", "")
        if quality in CLEAN_QUALITY_LEVELS:
            transcripts[data["call_id"]] = data
    return transcripts


def load_results(results_path: Path) -> list:
    """Load the experiment results JSON array."""
    with open(results_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ask_llm_match(
    planted_violations: list[str],
    evidence: str,
    reason: str,
) -> str:
    """Ask the LLM whether a detected violation matches any planted violation.

    Returns the exact planted violation text that matched, or 'NO_MATCH'.
    Retries up to MAX_RETRIES times on connection errors only.
    """
    planted_list = "\n".join(f"  - {v}" for v in planted_violations)
    prompt = (
        "Does this detected violation describe the same agent behavior "
        "as any of these planted violations?\n\n"
        f"Planted violations:\n{planted_list}\n\n"
        f"Detected violation evidence: {evidence}\n"
        f"Detected violation reason: {reason}\n\n"
        "Reply with the exact matching planted violation text, "
        "or 'NO_MATCH' if none match."
    )

    import time
    for attempt in range(1, 10):
        try:
            time.sleep(2.0)  # Pacing to respect the 30 RPM rate limit
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if "rate_limit" in str(exc).lower() or "429" in str(exc) or (
                hasattr(exc, "status_code") and exc.status_code == 429
            ):
                sleep_time = min(60, 2 ** attempt * 5)
                print(f"    [!] Rate limit hit. Sleeping {sleep_time}s (attempt {attempt}/9)...")
                time.sleep(sleep_time)
            elif isinstance(exc, (ConnectionError, OSError, groq.APIConnectionError, groq.APITimeoutError)):
                if attempt == 9:
                    print(f"    [!] Connection failed after 9 retries: {exc}")
                    return "NO_MATCH"
                print(f"    [!] Connection error (attempt {attempt}): {exc}. Retrying…")
            else:
                raise exc


def match_violations(
    planted: list[str],
    detected: list[dict],
) -> tuple[int, int, int, list[dict]]:
    """Match detected violations against planted violations.

    Returns (tp, fp, fn, match_details).
    """
    matched_planted: set[str] = set()
    match_details: list[dict] = []

    for det in detected:
        evidence = det.get("evidence", "")
        reason = det.get("reason", "")
        llm_answer = ask_llm_match(planted, evidence, reason)

        # Check if the LLM answer is one of the planted violations
        matched = None
        if llm_answer != "NO_MATCH":
            # Try exact match first
            for pv in planted:
                if pv == llm_answer:
                    matched = pv
                    break
            # Fallback: check if the LLM answer contains a planted violation
            if matched is None:
                for pv in planted:
                    if pv.lower() in llm_answer.lower():
                        matched = pv
                        break

        if matched and matched not in matched_planted:
            matched_planted.add(matched)
            match_details.append({
                "detected_evidence": evidence,
                "detected_reason": reason,
                "matched_planted": matched,
                "status": "TP",
            })
        else:
            match_details.append({
                "detected_evidence": evidence,
                "detected_reason": reason,
                "matched_planted": None,
                "llm_response": llm_answer,
                "status": "FP",
                "note": "duplicate match" if matched else "no match",
            })

    tp = len(matched_planted)
    fp = sum(1 for d in match_details if d["status"] == "FP")
    fn = len(planted) - tp

    return tp, fp, fn, match_details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python evaluate_gt.py <results_file>")
        print("  e.g. python evaluate_gt.py data/experiments/call_class_results.json")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    if not results_path.is_absolute():
        results_path = _PROJECT_ROOT / results_path

    if not results_path.exists():
        print(f"Error: results file not found: {results_path}")
        sys.exit(1)

    print(f"Results file : {results_path.name}")
    print(f"Transcripts  : {TRANSCRIPTS_DIR}")
    print(DIVIDER)

    # Load data
    transcripts = load_transcripts()
    results = load_results(results_path)
    results_by_id = {r["call_id"]: r for r in results}

    print(f"Transcripts with planted violations: {len(transcripts)}")
    print(f"Total results entries              : {len(results)}")
    print(DIVIDER)

    # Accumulators
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_call: list[dict] = []

    for call_id, transcript in sorted(transcripts.items()):
        planted = transcript.get("planted_violations", [])
        if not planted:
            continue

        result = results_by_id.get(call_id)
        if result is None:
            print(f"  [{call_id}] ⚠ No result found — skipping")
            fn_here = len(planted)
            total_fn += fn_here
            per_call.append({
                "call_id": call_id,
                "quality_level": transcript.get("quality_level"),
                "planted_count": len(planted),
                "detected_count": 0,
                "tp": 0,
                "fp": 0,
                "fn": fn_here,
                "note": "no result found",
            })
            continue

        detected = result.get("violations", [])
        print(f"  [{call_id}]  planted={len(planted)}  detected={len(detected)}")

        tp, fp, fn, details = match_violations(planted, detected)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_call.append({
            "call_id": call_id,
            "quality_level": transcript.get("quality_level"),
            "planted_count": len(planted),
            "detected_count": len(detected),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "match_details": details,
        })

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"           TP={tp}  FP={fp}  FN={fn}  "
              f"P={p:.2f}  R={r:.2f}  F1={f:.2f}")

    # --- Clean-call FP analysis ---
    clean_transcripts = load_clean_transcripts()
    clean_fp_calls = 0
    clean_fp_violations = 0
    clean_call_details = []

    for call_id, transcript in sorted(clean_transcripts.items()):
        result = results_by_id.get(call_id)
        if result is None:
            continue
        violations = result.get("violations", [])
        if violations:
            clean_fp_calls += 1
            clean_fp_violations += len(violations)
            clean_call_details.append({
                "call_id": call_id,
                "quality_level": transcript.get("quality_level"),
                "false_positive_violations": len(violations),
            })

    # Overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Print summary
    print()
    print(DIVIDER)
    print("  GROUND TRUTH EVALUATION SUMMARY")
    print(DIVIDER)
    print(f"  Results file    : {results_path.name}")
    print(f"  Calls evaluated : {len(per_call)}")
    print(f"  Total TP        : {total_tp}")
    print(f"  Total FP        : {total_fp}")
    print(f"  Total FN        : {total_fn}")
    print(f"  Precision       : {precision:.4f}")
    print(f"  Recall          : {recall:.4f}")
    print(f"  F1              : {f1:.4f}")
    print()
    print(f"  --- Clean-Call False Positive Rate ---")
    print(f"  Clean calls total   : {len(clean_transcripts)}")
    print(f"  Clean calls flagged : {clean_fp_calls}")
    print(f"  Clean call FP rate  : {clean_fp_calls / len(clean_transcripts):.2%}" if len(clean_transcripts) > 0 else "  Clean call FP rate  : 0.00%")
    print(f"  Total FP violations : {clean_fp_violations}")
    print(DIVIDER)

    # Save output
    output = {
        "results_file": results_path.name,
        "calls_evaluated": len(per_call),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_call": per_call,
        "clean_call_fp_rate": {
            "clean_calls_total": len(clean_transcripts),
            "clean_calls_flagged": clean_fp_calls,
            "fp_rate": round(clean_fp_calls / len(clean_transcripts), 4) if clean_transcripts else 0.0,
            "total_fp_violations": clean_fp_violations,
            "details": clean_call_details,
        },
    }

    stem = results_path.stem
    output_path = EXPERIMENTS_DIR / f"gt_eval_{stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nResults saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
