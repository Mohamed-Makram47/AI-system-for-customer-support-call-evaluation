"""
main.py — Wires classifier + RAG evaluator over test transcripts.
"""
import json
from pathlib import Path

from src.classifier import ClassifierPipeline
from src.runtime_rag import RAGEvaluator
from src.transcripts import TRANSCRIPTS
from src.scoring import score_all_calls

DIVIDER      = "=" * 70
RESULTS_PATH = Path("data/results.json")


def main():
    print(DIVIDER)
    print("  Phase 4 RAG — Compliance Evaluator")
    print(DIVIDER)

    print("\n[1/2] Loading classifier...")
    classifier = ClassifierPipeline()

    print("\n[2/2] Loading RAG evaluator...")
    evaluator = RAGEvaluator()

    print(f"\nLoaded. Running {len(TRANSCRIPTS)} transcripts.\n")

    total_violations = 0
    all_results      = []

    for transcript in TRANSCRIPTS:
        call_id        = transcript["call_id"]
        expected_label = transcript["fine_label"]
        utterances     = transcript["utterances"]

        # ── Classify: combine all customer turns ──────────────────────────
        customer_text    = " ".join(
            u["text"] for u in utterances if u["speaker"] == "customer"
        )
        prediction       = classifier(customer_text)
        predicted_label  = prediction["fine_label"]
        classifier_match = predicted_label == expected_label
        match_icon       = "OK" if classifier_match else "MISMATCH"

        print(DIVIDER)
        print(f"  Call ID  : {call_id}")
        print(f"  Expected : {expected_label}")
        print(f"  Predicted: {predicted_label}  [{match_icon}]")
        print(DIVIDER)

        agent_turns = [u["text"] for u in utterances if u["speaker"] == "agent"]
        result      = evaluator.evaluate_call(agent_turns, predicted_label)

        print(f"  Verdict  : {result['verdict'].upper()}")
        print(f"  Summary  : {result['overall_summary']}")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Rules used: {result['rules_retrieved']}")
        if result["violations"]:
            for v in result["violations"]:
                print(f"\n  !! VIOLATION (turn {v.get('turn')})")
                print(f"     Policy  : {v.get('violated_policy')}")
                print(f"     Evidence: {v.get('evidence')}")
                print(f"     Reason  : {v.get('reason')}")

        if result["verdict"] == "violation":
            total_violations += 1

        all_results.append({
            "call_id":              call_id,
            "fine_label_expected":  expected_label,
            "fine_label_predicted": predicted_label,
            "classifier_match":     classifier_match,
            "verdict":              result["verdict"],
            "violations":           result["violations"],
            "overall_summary":      result["overall_summary"],
            "confidence":           result["confidence"],
        })

        print()

    # ── Summary ───────────────────────────────────────────────────────────
    print(DIVIDER)
    print("  SUMMARY")
    print(DIVIDER)
    print(f"  Transcripts evaluated : {len(TRANSCRIPTS)}")
    print(f"  Calls with violations : {total_violations}")
    print(f"  Compliance rate       : "
          f"{((len(TRANSCRIPTS) - total_violations) / len(TRANSCRIPTS) * 100):.1f}%")
    print(DIVIDER)

    # ── Save results ──────────────────────────────────────────────────────
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Results saved to: {RESULTS_PATH.resolve()}")

    print("\n[Scoring] Computing call quality scores...")
    score_all_calls("data/results.json", TRANSCRIPTS)


if __name__ == "__main__":
    main()
