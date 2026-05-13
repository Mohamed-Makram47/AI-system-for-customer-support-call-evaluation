"""
run_real_call.py — Evaluates a real multi-issue call using automatic topic segmentation.

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
from src.topic_segmenter import segment_transcript, print_segments

DIVIDER = "=" * 70

# Filename on disk has a trailing space
DATA_PATH    = Path(__file__).parent.parent / "test_data" / "real_call_analysis .json"
RESULTS_PATH = Path(__file__).parent / "data" / "real_call_results.json"


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print(DIVIDER)
    print("  Real Call Evaluator - Phase 4 + 5")
    print(DIVIDER)

    if not DATA_PATH.exists():
        print(f"[ERROR] File not found: {DATA_PATH}")
        sys.exit(1)

    raw       = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    all_turns = raw["transcript"]
    print(f"\nLoaded {len(all_turns)} turns  |  duration: {raw['duration_seconds']:.1f}s\n")

    # ── Load models once ──────────────────────────────────────────────────────
    print("[1/2] Loading classifier...")
    classifier = ClassifierPipeline()

    print("\n[2/2] Loading RAG evaluator...")
    evaluator = RAGEvaluator()

    # ── Detect topic shifts automatically ─────────────────────────────────────
    print("\n[Topic Segmentation] Detecting topic shifts...")
    segments = segment_transcript(
        turns=all_turns,
        embedder=evaluator.embedder,
        min_words=8,
        threshold=0.50,
    )
    print_segments(segments)

    # ── Evaluate each segment ─────────────────────────────────────────────────
    all_results = []

    for seg in segments:
        agent_turns    = seg["agent_turns"]
        customer_turns = seg["customer_turns"]
        customer_text  = " ".join(customer_turns)
        seg_id         = f"REAL-Segment-{seg['segment_id']}"

        print(f"\n{DIVIDER}")
        print(f"  {seg_id}  |  {seg['start_time']:.1f}s - {seg['end_time']:.1f}s")
        print(f"  Turns total: {len(seg['turns'])}  |  agent: {len(agent_turns)}  |  customer: {len(customer_turns)}")
        print(DIVIDER)

        # Classify this segment
        if not customer_text.strip():
            print("  [SKIP] No customer turns in segment.")
            continue

        prediction      = classifier(customer_text)
        predicted_label = prediction["fine_label"]
        print(f"\n  Classifier -> {predicted_label}")

        # RAG evaluation using predicted label
        if not agent_turns:
            print("  [SKIP] No agent turns to evaluate.")
            continue

        rag_result = evaluator.evaluate_call(agent_turns, predicted_label)
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
            "call_id":              seg_id,
            "fine_label_expected":  predicted_label,
            "fine_label_predicted": predicted_label,
            "classifier_match":     True,
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
    print(f"  {'Segment':<18} {'Compliance':>10} {'Resolution':>11} {'Comm':>6} {'Final':>7} {'Grade':>6}")
    print("  " + "-" * 57)
    for r in all_results:
        s = r["score"]
        print(
            f"  {r['call_id']:<18}"
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
