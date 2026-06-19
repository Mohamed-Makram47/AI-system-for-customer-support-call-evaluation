"""
Reusable call-level evaluation function.
"""
import json
import time

from config import GROQ_EXPERIMENT_MODEL as GROQ_MODEL

def evaluate(utterances, fine_label, policies_text, client):
    """
    Evaluates the full conversation by sending interleaved utterances
    and the full policy manual text to the Groq LLM model in a single prompt.

    Args:
        utterances (list[dict]): A list of all dialogue turns (both agent and customer).
        fine_label (str): The predicted fine intent category.
        policies_text (str): The retrieved policies text.
        client: The Groq client.

    Returns:
        dict: Evaluation results containing verdict, violations list, and overall summary.
    """
    # Build interleaved conversation labeled with speaker role
    conversation = "\n".join(
        f"[{i+1}] {u['speaker'].capitalize()}: \"{u['text']}\""
        for i, u in enumerate(utterances)
    )

    # Single LLM call configuration and prompt creation
    prompt = (
        f"You are a banking compliance checker.\n"
        f"Issue class: {fine_label}\n\n"
        f"CONVERSATION:\n{conversation}\n\n"
        f"POLICIES:\n{policies_text}\n\n"
        f"Your task: Check if the agent violated any of the policies listed above.\n"
        f"For each violation found, return the exact policy rule code (e.g. cancel_transfer:R1 or baseline:B2).\n\n"
        f"Rules:\n"
        f"- ONLY flag violations of policies explicitly listed above\n"
        f"- Do NOT flag general customer service issues not covered by the policies\n"
        f"- If the agent made a mistake early but fully corrected it before the call ended, do NOT flag it\n\n"
        #===============================================
        f"Reply ONLY in this JSON format:\n"
        f'{{"verdict": "violation" or "ok", '
        f'"recovered": true or false, '
        f'"recovery_note": "one sentence or empty string", '
        f'"violations": [{{"turn": 1, "violated_policy": "exact_rule_code e.g. cancel_transfer:R1", '
        f'"evidence": "exact quote from conversation", "reason": "one sentence"}}]}}'
    )

    # Groq API rate-limit resilient retry loop (max 9 attempts)
    response = None
    for attempt in range(1, 10):
        try:
            time.sleep(2.0)  # Pacing to respect the 30 RPM rate limit
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            break
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e) or (
                hasattr(e, "status_code") and e.status_code == 429
            ):
                sleep_time = min(60, 2 ** attempt * 5)
                print(f"  [Groq] Rate limit hit. Sleeping {sleep_time}s (attempt {attempt}/9)...")
                time.sleep(sleep_time)
            else:
                raise e

    if response is None:
        print(f"  [Groq-CallLevel] All attempts failed. Skipping call.")
        return {
            "verdict":         "error",
            "recovered":       False,
            "recovery_note":   "",
            "violations":      [],
            "overall_summary": "All API attempts failed.",
        }

    # Extract and parse JSON robustly from the LLM response
    raw = response.choices[0].message.content.strip()
    try:
        start = raw.index("{")
        end   = raw.rindex("}") + 1
        parsed = json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        parsed = {
            "verdict": "error",
            "violations": [],
            "overall_summary": raw,
        }

    # Return evaluation result
    return {
        "verdict":         parsed.get("verdict"),
        "recovered":       parsed.get("recovered", False),
        "recovery_note":   parsed.get("recovery_note", ""),
        "violations":      parsed.get("violations", []),
        "overall_summary": parsed.get("overall_summary", ""),
    }
