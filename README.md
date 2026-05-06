# AI System for Customer Support Call Evaluation

An end-to-end agentic AI system that automatically evaluates customer support calls and generates personalized coaching feedback for agents — replacing manual, subjective quality assurance with a fully automated, explainable pipeline.

---

## The Problem

Current customer support evaluation suffers from:
- Manual quality assurance that is costly and slow
- Subjective and inconsistent scoring
- Limited call sampling — only a fraction of calls get reviewed
- No automatic policy compliance verification
- Lack of structured coaching for agents

---

## What the System Does

When a call ends, the system automatically:

1. **Transcribes** the call audio and identifies who is speaking (agent vs customer)
2. **Extracts behavioral metrics** — talking ratio, interruptions, silences, number of turns
3. **Evaluates the call using deep learning:**
   - Customer sentiment (Positive / Neutral / Negative) across the call
   - Agent opening and closing compliance
   - Issue type classification (78 banking categories)
4. **Validates agent resolution** against company policy using RAG (Retrieval-Augmented Generation)
5. **Computes a final performance score** using weighted metrics
6. **Generates a coaching report** with strengths, areas for improvement, and suggested alternative phrasing

---

## System Pipeline

```
Call Recording
      ↓
Phase 1 — Speech-to-Text + Speaker Diarization (Whisper + pyannote.audio)  ✅ Done
      ↓
Phase 2 — Rule-Based Behavioral Metrics (Talking ratio, interruptions, silences)  🔄 In Progress
      ↓
Phase 3 — Deep Learning Evaluation
      ├── 3A: Customer Sentiment (RoBERTa — 3 class)                        🔄 In Progress
      ├── 3B: Agent Opening/Closing Compliance (RoBERTa — binary)           ⬜ Planned
      └── 3C: Issue Classification (DualHead RoBERTa-large + SBERT ensemble — 10 coarse / 78 fine) ✅ Done
      ↓
Phase 4 — RAG Policy Validation (FAISS + sentence-transformers + Groq LLM)  ✅ Done
      ↓
Phase 5 — Scoring + Coaching Generation (LLM)                               ✅ Done
      ↓
Phase 6 — Web Dashboard (React / Streamlit)                                  ⬜ Planned
```

---

## Current Progress

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 3C** | Issue Classification — Dual-head RoBERTa-large + SBERT ensemble fine-tuned on BANKING77 (10 coarse + 78 fine classes, ~94% accuracy) | ✅ Done |
| **Phase 4** | RAG Policy Validation — 78 FAISS indexes (one per issue type), retrieves top-k policy rules and evaluates agent compliance via Groq LLM | ✅ Done |
| **Phase 5** | Scoring + Coaching Generation — LLM generates per-call coaching reports with violations, strengths, and suggested alternative phrasing | ✅ Done |
| **Phase 1** | Speech-to-Text & Speaker Diarization (Whisper + pyannote.audio) | 🔄 In Progress |
| **Phase 3A** | Customer Sentiment Analysis (RoBERTa — 3 class) | 🔄 In Progress |
| **Phase 3B** | Agent Opening/Closing Compliance (RoBERTa — binary) | ⬜ Planned |
| **Phase 2** | Rule-Based Behavioral Metrics (talking ratio, interruptions, silences) | ⬜ Planned |
| **Phase 6** | Web Dashboard (React / Streamlit) | ⬜ Planned |

### Completed Highlights

**Phase 3C — Issue Classification**
- Architecture: Dual-head RoBERTa-large with SBERT fallback ensemble (`all-mpnet-base-v2`)
- Coarse head: 10 banking categories | Fine head: 78 issue classes
- Accuracy: 93.9% on BANKING77 test set
- Model on HuggingFace: [Mohamed-Makram47/banking-issue-classifier](https://huggingface.co/Mohamed-Makram47/banking-issue-classifier)

**Phase 4 — RAG Policy Validation**
- 78 FAISS indexes pre-built, one per issue type
- Manuals grounded in real Banking77 customer utterances (10 examples per class)
- Embedding model: `all-MiniLM-L6-v2` (sentence-transformers)
- Call-level evaluation: single LLM call per conversation with deduplicated policy retrieval

**Phase 5 — Scoring + Coaching Generation**
- Aggregates RAG verdicts into a per-call compliance score
- LLM generates structured coaching reports per call:
  - Policy violations with explanation
  - Agent strengths
  - Suggested alternative phrasing for failed interactions

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Speech-to-Text | OpenAI Whisper |
| Speaker Diarization | pyannote.audio |
| NLP Models | RoBERTa (PyTorch + HuggingFace) |
| Issue Classification | Dual-head RoBERTa-large + SBERT ensemble (10 coarse + 78 fine classes) |
| RAG | FAISS + sentence-transformers (`all-MiniLM-L6-v2`) |
| LLM | Groq API (Llama 3) |
| Backend | Python + FastAPI |
| Frontend | React |
| Database | PostgreSQL + Vector DB |

---
