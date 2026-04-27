"""
src/coaching.py — Generates per-call coaching reports from results.json.
"""
import json
from pathlib import Path

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL

COACHING_DIR = Path("data/coaching")


def generate_coaching_report(call_result: dict) -> dict:
    fine_label  = call_result["fine_label_predicted"]
    agent_turns = call_result["agent_turns"]

    violations  = [t for t in agent_turns if t["verdict"] == "violation"]
    good_turns  = [t for t in agent_turns if t["verdict"] == "ok"]

    # ── Format good turns ─────────────────────────────────────────────────
    if good_turns:
        good_text = "\n".join(f'- "{t["utterance"]}"' for t in good_turns)
    else:
        good_text = "- (none)"

    # ── Format violations ─────────────────────────────────────────────────
    if violations:
        violation_lines = []
        for t in violations:
            violation_lines.append(
                f'- Utterance : "{t["utterance"]}"\n'
                f'  Policy    : {t["violated_policy"]}\n'
                f'  Reason    : {t["reason"]}'
            )
        violation_text = "\n".join(violation_lines)
    else:
        violation_text = "- (none)"

    # ── Build prompt ──────────────────────────────────────────────────────
    prompt = (
        f"You are a banking call center coach. Based on this agent's performance, "
        f"write a coaching report.\n\n"
        f"Issue class handled: {fine_label}\n\n"
        f"GOOD responses (agent did well):\n{good_text}\n\n"
        f"VIOLATIONS found:\n{violation_text}\n\n"
        f"Write a coaching report with exactly these three sections:\n"
        f"1. STRENGTHS: 2-3 bullet points of what the agent did well\n"
        f"2. IMPROVEMENTS: 2-3 bullet points of specific behaviors to fix\n"
        f"3. REPHRASING: For each violation, write one better alternative phrasing "
        f"the agent should have used instead\n\n"
        f"Reply ONLY in this JSON format:\n"
        f'{{"strengths": ["...", "..."], '
        f'"improvements": ["...", "..."], '
        f'"rephrasing": [{{"original": "...", "better": "..."}}]}}'
    )

    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "strengths":    ["(parse error — raw response below)"],
            "improvements": [],
            "rephrasing":   [],
            "raw":          raw,
        }

    return {
        "call_id":              call_result["call_id"],
        "fine_label_predicted": fine_label,
        "classifier_match":     call_result["classifier_match"],
        "violations_count":     len(violations),
        "good_turns_count":     len(good_turns),
        "strengths":            parsed.get("strengths", []),
        "improvements":         parsed.get("improvements", []),
        "rephrasing":           parsed.get("rephrasing", []),
    }


def generate_all_reports(results_path: str = "data/results.json") -> None:
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))

    COACHING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating coaching reports for {len(results)} calls...\n")

    for call_result in results:
        call_id = call_result["call_id"]
        print(f"  {call_id} ...", end=" ", flush=True)

        report      = generate_coaching_report(call_result)
        output_path = COACHING_DIR / f"{call_id}.json"
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"OK  ({report['violations_count']} violations, "
              f"{report['good_turns_count']} good turns)")

    print(f"\nDone. Reports saved to: {COACHING_DIR.resolve()}")


if __name__ == "__main__":
    generate_all_reports()
