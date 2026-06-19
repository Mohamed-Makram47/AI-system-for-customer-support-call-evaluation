"""
add_scores.py — Adds score objects to call_class_gt_results.json
Run from phase4_5_rag_coaching/ directory:
    C:\v311\Scripts\python add_scores.py
"""
import sys
sys.path.insert(0, ".")

from scripts.transcripts import TRANSCRIPTS
from src.scoring import score_all_calls

if __name__ == "__main__":
    print("Adding scores to call_class_gt_results.json...")
    score_all_calls(
        "data/experiments/call_class_gt_results.json",
        TRANSCRIPTS
    )
