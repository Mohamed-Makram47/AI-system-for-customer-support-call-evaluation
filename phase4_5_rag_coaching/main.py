"""
main.py — Wires classifier + RAG evaluator over test transcripts.
"""
import json
from pathlib import Path

from src.classifier import ClassifierPipeline
from src.runtime_rag import RAGEvaluator
from src.transcripts import TRANSCRIPTS

DIVIDER      = "=" * 70
SUB_DIVIDER  = "-" * 50
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

    total_agent_turns = 0
    total_violations  = 0
    all_results       = []

    for transcript in TRANSCRIPTS:
        call_id         = transcript["call_id"]
        expected_label  = transcript["fine_label"]
        utterances      = transcript["utterances"]

        # ── Classify: combine all customer turns ──────────────────────────
        customer_text   = " ".join(
            u["text"] for u in utterances if u["speaker"] == "customer"
        )
        prediction      = classifier(customer_text)
        predicted_label = prediction["fine_label"]
        classifier_match = predicted_label == expected_label
        match_icon      = "OK" if classifier_match else "MISMATCH"

        print(DIVIDER)
        print(f"  Call ID  : {call_id}")
        print(f"  Expected : {expected_label}")
        print(f"  Predicted: {predicted_label}  [{match_icon}]")
        print(DIVIDER)

        eval_label   = predicted_label
        agent_turns  = [u for u in utterances if u["speaker"] == "agent"]
        turns_output = []

        for i, turn in enumerate(agent_turns, 1):
            utterance = turn["text"]
            total_agent_turns += 1

            print(f"\n  Agent turn {i}: \"{utterance}\"")
            print(SUB_DIVIDER)

            result       = evaluator.evaluate(utterance, eval_label)
            verdict      = result["verdict"]
            is_violation = verdict == "violation"

            if is_violation:
                total_violations += 1

            verdict_icon = "!! VIOLATION" if is_violation else "   ok"
            print(f"  Verdict  : {verdict_icon}")
            print(f"  Reason   : {result['reason']}")

            if is_violation:
                print(f"  Policy   : {result['violated_policy']}")
                print(f"  Evidence : {result['evidence']}")

            print(f"  Confidence     : {result['confidence']}")
            print(f"  Retrieved rules: {len(result['retrieved_rules'])}")

            turns_output.append({
                "turn":            i,
                "utterance":       utterance,
                "verdict":         result["verdict"],
                "violated_policy": result["violated_policy"],
                "reason":          result["reason"],
                "evidence":        result["evidence"],
                "confidence":      result["confidence"],
                "retrieved_rules": result["retrieved_rules"],
            })

        all_results.append({
            "call_id":              call_id,
            "fine_label_expected":  expected_label,
            "fine_label_predicted": predicted_label,
            "classifier_match":     classifier_match,
            "agent_turns":          turns_output,
        })

        print()

    # ── Summary ───────────────────────────────────────────────────────────
    print(DIVIDER)
    print("  SUMMARY")
    print(DIVIDER)
    print(f"  Transcripts evaluated : {len(TRANSCRIPTS)}")
    print(f"  Agent turns evaluated : {total_agent_turns}")
    print(f"  Violations found      : {total_violations}")
    print(f"  Compliance rate       : "
          f"{((total_agent_turns - total_violations) / total_agent_turns * 100):.1f}%")
    print(DIVIDER)

    # ── Save results ──────────────────────────────────────────────────────
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Results saved to: {RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
