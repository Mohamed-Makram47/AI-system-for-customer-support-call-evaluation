"""
Experiment: Utterance-level evaluation + Class-scoped index.
Condition 3 of the 2x2 ablation matrix.
Saves results to data/experiments/utterance_class_results.json
"""
import sys
from pathlib import Path

# Adjust system path to import modules from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from scripts.transcripts import TRANSCRIPTS
from src.classifier import ClassifierPipeline
from scripts.retrievers import class_scoped
from scripts.evaluators import utterance_level
from scripts.classify_transcript import classify
from scripts.experiment_utils import print_summary, save_results, build_result, DIVIDER

# Define experiment directory and output file paths
EXPERIMENTS_DIR  = Path("data/experiments")
RESULTS_PATH     = EXPERIMENTS_DIR / "utterance_class_results.json"

def main():
    print(DIVIDER)
    print("  Experiment 3: Utterance-Level + Class-Scoped Index")
    print(DIVIDER)

    # Load components
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

        # 1) Retrieve policies using class-scoped
        try:
            policies = class_scoped.load(predicted_label)
        except FileNotFoundError:
            policies = "(no relevant policies retrieved above threshold)"

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
