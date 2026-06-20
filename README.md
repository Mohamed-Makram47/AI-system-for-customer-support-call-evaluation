# Automated Customer Support Call Evaluation System

An automated, intent-aware system for evaluating customer support calls. It converts recorded calls into structured evaluations, checks whether agents followed the relevant policies, calculates a quality score, and generates personalized coaching feedback.

The system is designed to support quality assurance teams by making call evaluation faster, more consistent, explainable, and scalable.

---

## Overview

Traditional call-center quality assurance depends on supervisors manually reviewing a small sample of recorded calls. This process is time-consuming, subjective, and unable to cover the full call volume.

This project automates the main stages of call evaluation:

* Transcribing recorded calls
* Identifying the agent and customer
* Measuring conversational behavior
* Detecting the customer’s banking issue
* Retrieving the policies relevant to that issue
* Evaluating agent compliance
* Calculating a final quality score
* Generating structured coaching feedback

---

## System Pipeline

```text
Call Recording
      │
      ▼
Phase 1 — Audio Processing
Speech-to-Text + Speaker Diarization
      │
      ├──────────────────────────────┐
      ▼                              ▼
Phase 2                        Phase 3
Behavioral Metrics             Banking Intent Classification
      │                              │
      └──────────────┬───────────────┘
                     ▼
Phase 4 — Final Call Evaluation
Topic Segmentation
        ↓
Policy Retrieval
        ↓
LLM Compliance Evaluation
        ↓
Quality Scoring
        ↓
Coaching Report
```

Phase 2 and Phase 3 operate as parallel branches after the transcript is produced by Phase 1.

---

## Phase 1 — Audio-to-Text and Speaker Diarization

The first phase converts a recorded customer support call into a structured, timestamped transcript.

The pipeline:

1. Converts the input audio into a standard 16 kHz mono format
2. Applies noise reduction
3. Transcribes the call using Whisper
4. Identifies speaker intervals using `pyannote.audio`
5. Merges transcription segments with speaker timestamps
6. Maps speakers to the functional roles of Agent and Customer
7. Detects long silence periods

The output contains:

* Transcript text
* Speaker labels
* Start and end timestamps
* Call duration
* Detected silence intervals
* Speaker and segment statistics

---

## Phase 2 — Timestamp-Based Behavioral Metrics

The second phase measures conversational behavior directly from the speaker timestamps produced by Phase 1.

It is completely rule-based and does not use a deep-learning model.

The extracted metrics include:

* **Talking ratio:** Percentage of the call occupied by each speaker
* **Interruption ratio:** Cases where the agent starts speaking before the customer finishes
* **Silence ratio:** Percentage of the call containing long silence periods
* **Number of turns:** Number of speaker changes during the conversation
* **Speaking duration:** Total speaking time for the agent and customer

These measurements provide transparent and reproducible indicators of communication quality.

---

## Phase 3 — Banking Intent Classification

The third phase identifies the main reason for the customer’s call.

Only customer turns are processed. Each turn is classified independently, and the informative predictions are combined to determine the primary call intent.

The classifier supports:

* 77 banking intents from the BANKING77 dataset
* 1 neutral category for greetings, acknowledgements, names, thanks, and other non-informative turns
* 10 broader banking categories used to support hierarchical classification

The classification architecture combines:

* Hierarchical BERT classification
* Coarse-category prediction
* Fine-grained intent prediction
* Sentence-BERT semantic embeddings
* Intent-centroid similarity
* Weighted BERT–SBERT fusion
* Call-level prediction aggregation

Neutral turns are excluded before the final call intent is selected, preventing greetings and conversational fillers from affecting the result.

---

## Phase 4 — Policy Compliance, Scoring, and Coaching

Phase 4 is the complete final evaluation stage.

It combines:

* The speaker-attributed transcript
* The conversational metrics
* The predicted banking intent
* The relevant policy rules

### Topic Segmentation

A call may contain more than one banking issue.

The system analyzes consecutive customer utterances using sentence embeddings and detects possible topic changes based on semantic similarity.

When a topic shift is detected, the call is divided into segments. Each segment can then be classified and evaluated independently.

### Policy Retrieval

The system uses Retrieval-Augmented Generation to retrieve the policy rules relevant to the detected intent.

Each banking intent has its own class-scoped FAISS index. Searching only the index associated with the detected intent reduces unrelated policy results and keeps the evaluation focused.

The retrieval component uses:

* FAISS vector search
* `all-MiniLM-L6-v2` sentence embeddings
* Cosine-similarity retrieval
* Similarity thresholding
* Top-k policy-rule selection

### Compliance Evaluation

The retrieved policy rules and the complete conversation are passed to a large language model.

The LLM evaluates whether each applicable rule was:

* Satisfied
* Partially satisfied
* Violated

For each detected problem, the system returns:

* The relevant policy rule
* The compliance decision
* Supporting transcript evidence
* An explanation of the decision

Using the complete call context allows the system to recognize recovery cases in which an agent makes an early mistake but later corrects it.

---

## Quality Scoring

The final score ranges from 0 to 100 and combines three components:

```text
Final Score =
    50% Policy Compliance
  + 30% Issue Resolution
  + 20% Communication Quality
```

### Policy Compliance

Measures how successfully the agent followed the retrieved policy rules.

* Satisfied rule: Full credit
* Partially satisfied rule: Half credit
* Violated rule: No credit

### Issue Resolution

Measures whether the customer’s issue was resolved or given an appropriate next step.

Possible outcomes include:

* Clearly resolved
* Resolved but dependent on an external action or waiting period
* Appropriately escalated
* Unresolved without a valid next step

### Communication Quality

Uses the Phase 2 measurements as its initial evidence.

It considers:

* Agent talking ratio
* Silence ratio
* Interruptions
* Turn-taking behavior
* Conversation context

The LLM may apply a limited contextual adjustment when unusual metrics have a reasonable explanation.

### Grades

| Grade |    Score |
| ----- | -------: |
| A     |   90–100 |
| B     |    80–89 |
| C     |    70–79 |
| D     | Below 70 |

The scoring weights, thresholds, and grade ranges can be adapted to the approved quality-assurance scorecard of a specific organization.

---

## Coaching Report

The system generates a structured coaching report for every evaluated call.

The report includes:

* Final numerical score
* Letter grade
* Policy compliance results
* Issue-resolution assessment
* Communication-quality assessment
* Agent strengths
* Confirmed violations
* Supporting transcript evidence
* Areas for improvement
* Suggested alternative phrasing

The goal is not only to identify mistakes, but also to provide actionable guidance that agents can use in future calls.

---

## Example Output

```json
{
  "call_id": "example_call",
  "primary_intent": "card_blocked",
  "behavioral_metrics": {
    "agent_talking_ratio": 0.54,
    "customer_talking_ratio": 0.36,
    "silence_ratio": 0.10,
    "interruptions": 2,
    "total_turns": 24
  },
  "policy_compliance_score": 85,
  "issue_resolution_score": 100,
  "communication_score": 90,
  "final_score": 90.5,
  "grade": "A",
  "strengths": [],
  "violations": [],
  "areas_for_improvement": [],
  "suggested_phrasing": []
}
```

---

## Core Technologies

| Component                      | Technology                          |
| ------------------------------ | ----------------------------------- |
| Audio preprocessing            | Python audio-processing tools       |
| Speech-to-text                 | Whisper / faster-whisper            |
| Speaker diarization            | pyannote.audio                      |
| Intent classification          | BERT                                |
| Semantic intent representation | Sentence-BERT / all-mpnet-base-v2   |
| Topic segmentation             | all-MiniLM-L6-v2                    |
| Policy retrieval               | FAISS                               |
| Compliance evaluation          | Groq-hosted LLM                     |
| Scoring and coaching           | Python + LLM                        |
| Model development              | PyTorch + Hugging Face Transformers |

---

## Intended Use

The system is designed as an intent-aware decision-support tool for customer support quality assurance.

It can help organizations:

* Review a larger percentage of their calls
* Apply more consistent evaluation criteria
* Detect intent-specific policy violations
* Identify unresolved customer issues
* Provide evidence-based coaching
* Focus human reviewers on calls requiring further attention

The policy manuals used by the research prototype are project-specific research artifacts.

Deployment within a real organization requires replacing them with approved company policies and validating the complete system with qualified quality-assurance and compliance specialists.
