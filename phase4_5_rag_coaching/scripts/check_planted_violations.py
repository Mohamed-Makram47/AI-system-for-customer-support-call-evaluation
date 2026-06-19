import json
import os
import time
from pathlib import Path
from groq import Groq

import sys
base_dir = Path("phase4_5_rag_coaching")
sys.path.append(str(base_dir.resolve()))

try:
    from config import GROQ_API_KEY, GROQ_MODEL
except ImportError:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def main():
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not found in config or environment variables.")
        return

    client = Groq(api_key=GROQ_API_KEY)

    base_dir = Path("phase4_5_rag_coaching")
    transcripts_dir = base_dir / "data" / "transcripts"
    manuals_dir = base_dir / "manuals"
    
    baseline_path = manuals_dir / "baseline" / "baseline_policies.txt"
    if baseline_path.exists():
        baseline_rules = baseline_path.read_text(encoding="utf-8")
    else:
        baseline_rules = "No baseline rules found."

    eval_qualities = {"bad", "ambiguous", "incomplete"}
    
    total_violations = 0
    mapped_count = 0
    unmapped_count = 0

    for transcript_path in sorted(transcripts_dir.glob("*.json")):
        try:
            t = json.loads(transcript_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
            
        if t.get("quality_level") not in eval_qualities:
            continue
            
        planted_violations = t.get("planted_violations", [])
        if not planted_violations:
            continue
            
        intent = t.get("intent")
        intent_path = manuals_dir / f"{intent}.txt"
        
        if intent_path.exists():
            intent_rules = intent_path.read_text(encoding="utf-8")
        else:
            intent_rules = "No intent-specific rules found."
            
        print(f"\n[{t['call_id']}] {intent}")
        
        for violation in planted_violations:
            total_violations += 1
            
            prompt = f"""Given these policy rules:
{baseline_rules}
{intent_rules}

Does this planted violation map to any specific rule?
Planted violation: "{violation}"

Reply ONLY in this JSON format:
{{"maps_to": "rule_code e.g. baseline:B1 or cancel_transfer:R1", "confidence": "high/medium/low", "reason": "one sentence"}}
If no rule matches, return: {{"maps_to": "NONE", "confidence": "high", "reason": "one sentence explaining why"}}"""

            try:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                
                content = response.choices[0].message.content.strip()
                
                # Strip markdown codeblocks if the LLM adds them
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()
                    
                result = json.loads(content)
                maps_to = result.get("maps_to", "ERROR")
                confidence = result.get("confidence", "unknown")
                reason = result.get("reason", "")
                
                print(f"  VIOLATION: \"{violation}\"")
                print(f"  MAPS TO: {maps_to} ({confidence}) — {reason}")
                print()
                
                if maps_to.upper() == "NONE":
                    unmapped_count += 1
                elif maps_to != "ERROR":
                    mapped_count += 1
                    
            except Exception as e:
                print(f"  VIOLATION: \"{violation}\"")
                print(f"  ERROR: API call or JSON parsing failed: {e}\n  Raw content: {content if 'content' in locals() else 'None'}")
                print()
            
            # Respect rate limits (30 RPM on Groq free tier)
            time.sleep(2.5)

    print("\n" + "="*40)
    print("SUMMARY")
    print("="*40)
    print(f"Total violations checked: {total_violations}")
    print(f"Mapped: {mapped_count}")
    print(f"Unmapped (to remove): {unmapped_count}")

if __name__ == "__main__":
    main()
