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
Phase 1 — Speech-to-Text + Speaker Diarization (Whisper + pyannote.audio)
      ↓
Phase 2 — Rule-Based Behavioral Metrics (Talking ratio, interruptions, silences)
      ↓
Phase 3 — Deep Learning Evaluation
      ├── Customer Sentiment (RoBERTa — 3 class)
      ├── Agent Opening/Closing Compliance (RoBERTa — binary)
      └── Issue Classification (BERT — 78 classes) ✅ Done
      ↓
Phase 4 — RAG Policy Validation (FAISS + LLM)
      ↓
Phase 5 — Scoring + Coaching Generation (LLM)
      ↓
Dashboard (React / Streamlit)
```

---


---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Speech-to-Text | OpenAI Whisper |
| Speaker Diarization | pyannote.audio |
| NLP Models | BERT, RoBERTa (PyTorch + HuggingFace) |
| RAG | FAISS + sentence-transformers + Groq API |
| Coaching | Groq API (Llama 3) |
| Backend | Python + FastAPI |
| Frontend | React|
| Database | PostgreSQL + Vector DB |

---

## Current Progress

- ✅ Phase 3C — Issue Classification: BERT fine-tuned on BANKING77 (91.33% accuracy, 78 classes)
  - Model: [Mohamed-Makram77/banking-issue-classifier](https://huggingface.co/Mohamed-Makram47/banking-issue-classifier)
- 🔄 Phase 1 — Speech-to-Text & Diarization: In progress
- 🔄 Phase 3A — Sentiment Analysis: In progress
- ⬜ Phase 3B — Opening/Closing Compliance: Planned
- ⬜ Phase 4 — RAG Pipeline: Planned
- ⬜ Phase 5 — Scoring & Coaching: Planned
- ⬜ Phase 6 — Web Dashboard: Planned

---
