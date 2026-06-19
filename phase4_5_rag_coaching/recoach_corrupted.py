"""
recoach_corrupted.py — Re-generates coaching reports for corrupted files only.
Run from phase4_5_rag_coaching/ directory:
    C:\v311\Scripts\python recoach_corrupted.py
"""
import sys
sys.path.insert(0, ".")

import json
from pathlib import Path
from src.coaching import generate_coaching_report

CORRUPTED = [
    "cancel_transfer_bad_1",
    "wrong_amount_of_cash_received_bad_1",
]

RESULTS_PATH  = Path("data/experiments/call_class_gt_results.json")
COACHING_DIR  = Path("data/coaching")

def main():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    results_map = {r["call_id"]: r for r in results}

    for call_id in CORRUPTED:
        print(f"\n  Re-generating: {call_id} ...", end=" ", flush=True)

        call_result = results_map.get(call_id)
        if call_result is None:
            print(f"NOT FOUND in results — skipping.")
            continue

        report = generate_coaching_report(call_result)

        has_error = any(
            "parse error" in s
            for s in report.get("strengths", []) + report.get("improvements", [])
        )

        if has_error:
            print("PARSE ERROR AGAIN — not saved. Try re-running.")
            continue

        out_path = COACHING_DIR / f"{call_id}.json"
        out_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"OK  ({report['violations_count']} violations)")

    print("\nDone.")

if __name__ == "__main__":
    main()
