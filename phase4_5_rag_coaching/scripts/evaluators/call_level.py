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
        f"You are a banking call center QA evaluator.\n"
        f"Issue class: {fine_label}\n\n"
        f"CONVERSATION:\n{conversation}\n\n"
        f"POLICIES:\n{policies_text}\n\n"
        f"Evaluate the agent's FINAL performance in this conversation.\n"
        f"Focus on the outcome of the call, not individual turns.\n"
        f"If the agent made a mistake early but corrected it before the call ended, "
        f"do NOT flag it as a violation.\n"
        f"Only flag violations that were UNRESOLVED at the end of the call.\n\n"
        f"Reply ONLY in this JSON format:\n"
        f'{{"verdict": "violation" or "ok", '
        f'"recovered": true or false, '
        f'"recovery_note": "one sentence or empty string", '
        f'"violations": [{{"turn": 1, "violated_policy": "...", '
        f'"evidence": "...", "reason": "..."}}], '
        f'"overall_summary": "one sentence about the agent\'s overall performance"}}'
    )

    # Groq API rate-limit resilient retry loop (max 9 attempts)
    for attempt in range(1, 10):
        try:
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
                print(f"  [Groq] Rate limit hit. Sleeping 5s (attempt {attempt}/9)...")
                time.sleep(5)
            else:
                raise e

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
