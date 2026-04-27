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
    def evaluate(self, utterance: str, fine_label: str) -> dict:
        retrieved = self.retrieve(utterance, fine_label)

        if retrieved:
            rules_text = "\n".join(
                f"- {r['rule']}  [score: {r['score']}]" for r in retrieved
            )
            confidence = (
                "High" if any(r["score"] >= SIMILARITY_THRESHOLD for r in retrieved)
                else "Low"
            )
        else:
            rules_text = "(no relevant policies retrieved above threshold)"
            confidence = "Low"

        prompt = (
            f"You are a banking call center QA evaluator.\n"
            f"Issue class: {fine_label}\n"
            f"Agent said: \"{utterance}\"\n\n"
            f"Relevant policies:\n{rules_text}\n\n"
            f"Only return verdict: violation if the agent clearly and directly broke a "
            f"specific policy rule. If the agent's response is helpful, correct, or "
            f"neutral, return verdict: ok. When in doubt, return ok.\n\n"
            f"Reply ONLY in this JSON format:\n"
            f'{{"verdict": "violation" or "ok", '
            f'"violated_policy": "rule text or null", '
            f'"reason": "one sentence", '
            f'"evidence": "exact words from agent or null"}}'
        )

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if the model wraps the JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {
                "verdict":         "error",
                "violated_policy": None,
                "reason":          "LLM returned non-JSON response.",
                "evidence":        raw,
            }

        return {
            "verdict":          parsed.get("verdict"),
            "violated_policy":  parsed.get("violated_policy"),
            "reason":           parsed.get("reason"),
            "evidence":         parsed.get("evidence"),
            "confidence":       confidence,
            "retrieved_rules":  retrieved,
        }
