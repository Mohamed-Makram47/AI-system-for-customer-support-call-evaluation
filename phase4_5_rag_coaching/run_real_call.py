"""
run_real_call.py — Evaluates a real two-issue call against RAG policy + scoring.

Reads:  ../test_data/real_call_analysis .json   (note: space in filename)
Writes: data/real_call_results.json
"""
import json
import sys
from pathlib import Path

# ── make local src/ importable when run from this directory ──────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.classifier import ClassifierPipeline
from src.runtime_rag import RAGEvaluator
from src.scoring import compute_score

DIVIDER = "=" * 70

# Filename on disk has a trailing space
DATA_PATH    = Path(__file__).parent.parent / "test_data" / "real_call_analysis .json"
RESULTS_PATH = Path(__file__).parent / "data" / "real_call_results.json"

PARTS = [
    {"part_id": "REAL-Part-1", "label": "lost_or_stolen_card",      "threshold": (None, 130)},
    {"part_id": "REAL-Part-2", "label": "card_payment_fee_charged",  "threshold": (130, None)},
]


def split_transcript(turns: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split turns at start_time < 130 vs >= 130."""
    part1 = [t for t in turns if t["start_time"] <  130]
    part2 = [t for t in turns if t["start_time"] >= 130]
    return part1, part2


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print(DIVIDER)
    print("  Real Call Evaluator - Phase 4 + 5")
    print(DIVIDER)

    if not DATA_PATH.exists():
        print(f"[ERROR] File not found: {DATA_PATH}")
        sys.exit(1)

    raw        = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    all_turns  = raw["transcript"]
    print(f"\nLoaded {len(all_turns)} turns  |  duration: {raw['duration_seconds']:.1f}s\n")

    # ── Load models once ──────────────────────────────────────────────────────
    print("[1/2] Loading classifier...")
    classifier = ClassifierPipeline()

    print("\n[2/2] Loading RAG evaluator...")
    evaluator  = RAGEvaluator()

    # ── Split transcript ──────────────────────────────────────────────────────
    part1_turns, part2_turns = split_transcript(all_turns)
    turn_groups = [part1_turns, part2_turns]

    all_results = []

    for part_cfg, turns in zip(PARTS, turn_groups):
        part_id      = part_cfg["part_id"]
        fixed_label  = part_cfg["label"]

        agent_turns    = [t["text"] for t in turns if t["speaker"] == "Agent"]
        customer_turns = [t["text"] for t in turns if t["speaker"] == "Customer"]

        print(f"\n{DIVIDER}")
        print(f"  {part_id}  |  expected: {fixed_label}")
        print(f"  Turns total: {len(turns)}  |  agent: {len(agent_turns)}  |  customer: {len(customer_turns)}")
        print(DIVIDER)

        # Classify customer text
        customer_text   = " ".join(customer_turns)
        prediction      = classifier(customer_text)
        predicted_label = prediction["fine_label"]
        match_icon      = "OK" if predicted_label == fixed_label else "MISMATCH"
        print(f"\n  Classifier -> {predicted_label}  [{match_icon}]")

        # RAG evaluation — use fixed label so FAISS index is guaranteed
        rag_result = evaluator.evaluate_call(agent_turns, fixed_label)
        print(f"  Verdict    -> {rag_result['verdict'].upper()}")
        print(f"  Confidence -> {rag_result['confidence']}")
        print(f"  Summary    -> {rag_result['overall_summary']}")
        if rag_result["violations"]:
            for v in rag_result["violations"]:
                print(f"\n  !! VIOLATION (turn {v.get('turn')})")
                print(f"     Policy  : {v.get('violated_policy')}")
                print(f"     Evidence: {v.get('evidence')}")
                print(f"     Reason  : {v.get('reason')}")

        # Scoring
        call_result_dict = {
            "call_id":              part_id,
            "fine_label_expected":  fixed_label,
            "fine_label_predicted": predicted_label,
            "classifier_match":     predicted_label == fixed_label,
            "verdict":              rag_result["verdict"],
            "violations":           rag_result["violations"],
            "overall_summary":      rag_result["overall_summary"],
            "confidence":           rag_result["confidence"],
        }

        scoring = compute_score(call_result_dict, agent_turns)
        print(f"\n  Score      -> {scoring['final_score']} / 100  (Grade {scoring['grade']})")
        print(f"  Compliance -> {scoring['policy_compliance']}%")
        print(f"  Resolution -> {scoring['issue_resolution']['score']}  ({scoring['issue_resolution']['reason']})")
        print(f"  Comm       -> {scoring['communication']['score']}  ({scoring['communication']['note']})")

        call_result_dict["score"] = scoring
        all_results.append(call_result_dict)

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("  SUMMARY")
    print(DIVIDER)
    print(f"  {'Part':<14} {'Compliance':>10} {'Resolution':>11} {'Comm':>6} {'Final':>7} {'Grade':>6}")
    print("  " + "-" * 53)
    for r in all_results:
        s = r["score"]
        print(
            f"  {r['call_id']:<14}"
            f"{s['policy_compliance']:>10.1f}"
            f"{s['issue_resolution']['score']:>11}"
            f"{s['communication']['score']:>6}"
            f"{s['final_score']:>7.1f}"
            f"  {s['grade']}"
        )
    print(DIVIDER)
    print(f"\n  Results saved to: {RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
