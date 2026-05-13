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

## Project Structure

| Folder | Phase | Description | Status |
|--------|-------|-------------|--------|
| `phase1_audio_to_text/` | Phase 1 | Speech-to-text + speaker diarization (Whisper + pyannote) | ✅ Done |
| `issue_type_classification_model/` | Phase 3 | Issue classification — DualHead RoBERTa (78 classes) | ✅ Done |
| `phase4_5_rag_coaching/` | Phase 4 & 5 | RAG compliance evaluation + scoring + coaching | ✅ Done |

---

## End-to-End Pipeline Test

A real 4-minute 40-second banking call was recorded, transcribed through Phase 1,
classified through Phase 3, and evaluated through Phase 4 & 5.

- Real audio recorded by the team
- Two banking issues in one call: `lost_or_stolen_card` + `card_payment_fee_charged`
- Automatic topic shift detection using embedding similarity
- Self-correction correctly handled by call-level LLM evaluation
- Final scores: Segment 1–3 (B/C), Segment 4 (D — violation), Segment 5 (A — recovery)

Test files: `test_data/real_call_analysis.json` · `test_data/real_call_transcript.txt`

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
- 78 class-scoped FAISS indexes — one per fine issue class
- Policy manuals grounded in real Banking77 customer queries (10 examples per class)
- Embedding model: `all-MiniLM-L6-v2` (sentence-transformers, runs locally)
- Call-level evaluation: full conversation sent to LLM in one prompt
- Handles self-corrections — agent fixing a mistake is not penalised
- Output: violation verdict + violated policy + evidence + reason + confidence

**Phase 5 — Scoring + Coaching Generation**
- Weighted quality score (0–100): policy compliance 50% · issue resolution 30% · communication 20%
- LLM judges whether the agent successfully resolved the customer's issue
- Dismissive language detection for communication scoring
- Grade: A (90–100) · B (75–89) · C (60–74) · D (below 60)
- Coaching report per call: strengths · improvements · suggested rephrasing
- Output saved to `data/coaching/{call_id}.json`

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
