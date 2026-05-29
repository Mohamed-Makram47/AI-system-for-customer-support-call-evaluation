"""
Reusable utterance-level evaluation function.
Evaluates each customer+agent turn pair individually, then aggregates.
"""
import json
import time

from config import GROQ_EXPERIMENT_MODEL as GROQ_MODEL

def evaluate(utterances: list[dict], fine_label: str, policies_text: str, client) -> dict:
    """
    Evaluates the conversation utterance by utterance by sending each 
    customer-agent pair to the Groq LLM model.

    Args:
        utterances (list[dict]): A list of all dialogue turns.
        fine_label (str): The predicted fine intent category.
        policies_text (str): The retrieved policies text.
        client: The Groq client.

    Returns:
        dict: Evaluation results containing verdict, recovered, recovery_note, violations, and overall summary.
    """
    exchanges = []
    current_customer_text = None
    
    for u in utterances:
        spk = u.get("speaker", "").lower()
        text = u.get("text", "")
        if spk == "customer":
            current_customer_text = text
        elif spk == "agent":
            if current_customer_text is not None:
                exchanges.append({
                    "customer": current_customer_text,
                    "agent": text
                })
                current_customer_text = None

    all_violations = []
    overall_verdict = "ok"
    
    for i, exch in enumerate(exchanges):
        time.sleep(4)
        prompt = (
            f"You are a banking call center QA evaluator.\n"
            f"Issue class: {fine_label}\n\n"
            f"EXCHANGE:\nCustomer: \"{exch['customer']}\"\nAgent: \"{exch['agent']}\"\n\n"
            f"POLICIES:\n{policies_text}\n\n"
            f"Evaluate ONLY this single exchange. Did the agent violate any policy\n"
            f"in their response to this specific customer message?\n\n"
            f"Reply ONLY in this JSON format:\n"
            f'{{"verdict": "violation" or "ok", '
            f'"recovered": true or false, '
            f'"recovery_note": "one sentence or empty string", '
            f'"violations": [{{"turn": {i+1}, "violated_policy": "...", '
            f'"evidence": "...", "reason": "..."}}], '
            f'"overall_summary": "one sentence about the agent\'s overall performance"}}'
        )

        response = None
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
                    print(f"  [Groq] Rate limit hit. Sleeping 15s (attempt {attempt}/9)...")
                    time.sleep(15)
                else:
                    raise e
        if response is None:
            print(f"  [Groq-Utterance] All attempts failed for exchange {i+1}. Skipping.")
            continue

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

        if parsed.get("verdict") == "violation":
            overall_verdict = "violation"
            
        for v in parsed.get("violations", []):
            v["turn"] = i + 1
            all_violations.append(v)
            
    n = len(exchanges)
    v_count = len(all_violations)
    
    overall_summary = f"Evaluated {n} exchanges. {v_count} violation(s) found."
    
    return {
        "verdict": overall_verdict,
        "recovered": False,
        "recovery_note": "",
        "violations": all_violations,
        "overall_summary": overall_summary,
    }
