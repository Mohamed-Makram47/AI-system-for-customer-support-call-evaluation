"""
src/runtime_rag.py — Class-scoped RAG evaluator.
"""
import json
from pathlib import Path

import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer

from config import (
    GROQ_API_KEY, GROQ_MODEL,
    EMBEDDING_MODEL,
    INDEXES_DIR, MAPS_DIR,
    SIMILARITY_THRESHOLD, TOP_K,
)


class RAGEvaluator:

    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client   = Groq(api_key=GROQ_API_KEY)
        self._cache: dict = {}          # fine_label -> (index, policy_map)

    # ------------------------------------------------------------------
    def load_index(self, fine_label: str):
        if fine_label in self._cache:
            return self._cache[fine_label]

        index_path = Path(INDEXES_DIR) / f"{fine_label}.faiss"
        map_path   = Path(MAPS_DIR)    / f"{fine_label}.json"

        if not index_path.exists():
            raise FileNotFoundError(f"No FAISS index for class: {fine_label}")
        if not map_path.exists():
            raise FileNotFoundError(f"No policy map for class: {fine_label}")

        index      = faiss.read_index(str(index_path))
        policy_map = json.loads(map_path.read_text(encoding="utf-8"))

        self._cache[fine_label] = (index, policy_map)
        return index, policy_map

    # ------------------------------------------------------------------
    def retrieve(
        self,
        utterance: str,
        fine_label: str,
        k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> list[dict]:
        index, policy_map = self.load_index(fine_label)

        vec = self.embedder.encode([utterance], normalize_embeddings=True)
        vec = np.array(vec, dtype=np.float32)

        k_actual = min(k, index.ntotal)
        scores, indices = index.search(vec, k_actual)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) < threshold:
                continue
            entry = policy_map[idx]
            results.append({
                "rule":   entry["rule"],
                "source": entry["source"],
                "score":  round(float(score), 4),
            })

        return results

    # ------------------------------------------------------------------
    def evaluate_call(self, agent_turns: list[str], fine_label: str) -> dict:
        """
        Evaluates the full conversation at once.
        Retrieves policies for each agent turn, deduplicates, then sends
        ALL turns + ALL relevant policies to LLM in one call.
        Returns a single call-level verdict.
        """
        # Step 1: retrieve rules for every agent turn, deduplicate
        all_rules = {}
        for turn in agent_turns:
            retrieved = self.retrieve(turn, fine_label)
            for r in retrieved:
                all_rules[r["rule"]] = r["score"]  # dedup by rule text, keep highest score

        if all_rules:
            rules_text = "\n".join(f"- {rule}" for rule in all_rules.keys())
            confidence = "High" if max(all_rules.values()) >= SIMILARITY_THRESHOLD else "Low"
        else:
            rules_text = "(no relevant policies retrieved above threshold)"
            confidence = "Low"

        # Step 2: format all agent turns numbered
        turns_text = "\n".join(
            f"[{i+1}] \"{turn}\"" for i, turn in enumerate(agent_turns)
        )

        # Step 3: single LLM call for the whole conversation
        prompt = (
            f"You are a banking call center QA evaluator.\n"
            f"Issue class: {fine_label}\n\n"
            f"AGENT TURNS (full conversation):\n{turns_text}\n\n"
            f"RELEVANT POLICIES:\n{rules_text}\n\n"
            f"Evaluate the full conversation. Did the agent violate any policy?\n"
            f"Only flag a violation if the agent clearly and directly broke a rule "
            f"and did NOT correct themselves later in the conversation.\n\n"
            f"Reply ONLY in this JSON format:\n"
            f'{{"verdict": "violation" or "ok", '
            f'"violations": [{{"turn": 1, "violated_policy": "...", "evidence": "...", "reason": "..."}}], '
            f'"overall_summary": "one sentence about the agent\'s overall performance"}}'
        )

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {
                "verdict": "error",
                "violations": [],
                "overall_summary": raw,
            }

        return {
            "verdict":         parsed.get("verdict"),
            "violations":      parsed.get("violations", []),
            "overall_summary": parsed.get("overall_summary", ""),
            "confidence":      confidence,
            "rules_retrieved": len(all_rules),
        }
