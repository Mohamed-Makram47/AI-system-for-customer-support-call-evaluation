# AI System for Customer Support Call Evaluation 

## Phase 1: Audio-to-Text with Speaker Diarization

This repository contains the first phase of an AI-powered system designed for evaluating customer support calls. The primary goal of this phase is to accurately convert raw audio support calls into structured transcripts with speaker diarization.

### Contents

- **`phase1.ipynb`**: A Jupyter Notebook detailing the step-by-step experimentation, models utilized, and the logic built for testing the audio-to-text and diarization processes.
- **`Phase1pipeline.py`**: A clean, modular Python script containing the final pipeline function for Phase 1. It takes raw audio and returns a structured JSON transcript with identified speakers, ready to be passed to subsequent phases.

### Features
- Audio-to-text transcription.
- Speaker Diarization (distinguishing between customer and support agent).
- Prepares structured dictionary outputs containing transcripts, silences, and summaries for down-stream evaluation tasks.

### Setup and Usage

To use the `.py` pipeline directly in python code:

```python
from Phase1pipeline import run_phase1

result = run_phase1(
    audio_path = "path/to/call.wav",
    hf_token   = "your_hugging_face_token",
    model_size = "small"
)
```