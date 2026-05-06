"""
phase1_pipeline.py
==================
Phase 1: Audio-to-Text with Speaker Diarization

Role in the full project:
    This file is imported by the main project pipeline.
    It takes a raw audio file and returns a structured
    transcript JSON ready for Phase 2, 3, and 4.

Usage:
    from phase1_pipeline import run_phase1

    result = run_phase1(
        audio_path = "call.wav",
        hf_token   = "hf_xxx",
        model_size = "small"
    )
    # result is a dict with transcript, silences, summary

NOTE:
    Before importing this file, the caller must apply patches.
    In the full project main notebook, patches are applied in
    Cell 1 before this file is imported.
"""

import json
import os
import gc

import torch
import whisper
import librosa
import noisereduce as nr
import soundfile as sf
from pydub import AudioSegment
from pyannote.audio import Pipeline


# ─────────────────────────────────────────────────────────────────────
# FUNCTION 1 — Convert MP3 → WAV
# ─────────────────────────────────────────────────────────────────────
def convert_to_wav(path: str) -> str:
    """Convert MP3 to WAV if needed. Returns WAV path."""
    if path.lower().endswith(".mp3"):
        audio = AudioSegment.from_mp3(path)
        wav_path = path.replace(".mp3", ".wav")
        audio.export(wav_path, format="wav")
        return wav_path
    return path


# ─────────────────────────────────────────────────────────────────────
# FUNCTION 2 — Noise Reduction
# ─────────────────────────────────────────────────────────────────────
def denoise_audio(input_path: str) -> str:
    """
    Apply noise reduction to audio.
    Uses first 0.5s as noise profile.
    Returns path to denoised file.
    """
    data, rate = sf.read(input_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    noise_clip = data[:int(rate * 0.5)]
    reduced = nr.reduce_noise(y=data, sr=rate, y_noise=noise_clip, prop_decrease=0.8)
    output_path = input_path.replace(".wav", "_denoised.wav")
    sf.write(output_path, reduced, rate)
    return output_path


# ─────────────────────────────────────────────────────────────────────
# FUNCTION 3 — Whisper Transcription
# ─────────────────────────────────────────────────────────────────────
def run_transcription(audio_path: str, model_size: str = "small") -> tuple:
    """
    Transcribe audio using Whisper.
    Loads model to GPU, transcribes, then frees GPU memory.

    Args:
        audio_path: path to WAV file
        model_size: tiny / base / small / medium / large-v3
                    Use large-v3 for Arabic, small for English

    Returns:
        (segments, language) tuple
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = whisper.load_model(model_size, device=device)
    result = model.transcribe(audio_path, word_timestamps=True)
    lang   = result.get("language", "unknown")
    segs   = result["segments"]

    # Free GPU memory immediately
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return segs, lang


# ─────────────────────────────────────────────────────────────────────
# FUNCTION 4 — Speaker Diarization
# ─────────────────────────────────────────────────────────────────────
def run_diarization(audio_path: str, hf_token: str) -> list:
    """
    Run speaker diarization using pyannote.
    Loads pipeline to GPU, processes audio, then frees GPU memory.

    Args:
        audio_path: path to WAV file
        hf_token:   HuggingFace token for pyannote model access

    Returns:
        list of dicts with start, end, speaker keys
    """
    import torchaudio

    gc.collect()
    torch.cuda.empty_cache()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    # Pass waveform directly — more stable than path-based loading
    waveform, sample_rate = torchaudio.load(audio_path)

    with torch.no_grad():
        diarization = pipeline(
            {"waveform": waveform, "sample_rate": sample_rate},
            num_speakers=2
        )

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start":   round(turn.start, 2),
            "end":     round(turn.end,   2),
            "speaker": speaker
        })

    # Free GPU memory immediately
    del pipeline, waveform
    gc.collect()
    torch.cuda.empty_cache()

    return segments


# ─────────────────────────────────────────────────────────────────────
# FUNCTION 5 — Merge Whisper + Diarization
# ─────────────────────────────────────────────────────────────────────
def merge_transcript(whisper_segments: list, diarization_segments: list) -> list:
    """
    Assign each Whisper text segment the speaker with most overlap.
    """
    transcript = []
    for seg in whisper_segments:
        seg_start    = seg["start"]
        seg_end      = seg["end"]
        best_speaker = "Unknown"
        best_overlap = 0.0

        for d in diarization_segments:
            overlap = min(seg_end, d["end"]) - max(seg_start, d["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]

        transcript.append({
            "start_time": round(seg_start, 2),
            "end_time":   round(seg_end,   2),
            "speaker":    best_speaker,
            "text":       seg["text"].strip()
        })
    return transcript


# ─────────────────────────────────────────────────────────────────────
# FUNCTION 6 — Label Agent / Customer
# ─────────────────────────────────────────────────────────────────────
def label_speakers(transcript: list) -> list:
    """
    Map SPEAKER_00/01 → Agent/Customer.
    First speaker seen = Agent (agents always greet first).
    """
    label_map = {}
    for entry in transcript:
        spk = entry["speaker"]
        if spk not in label_map:
            label_map[spk] = "Agent" if len(label_map) == 0 else "Customer"
        entry["speaker"] = label_map[spk]
    return transcript


# ─────────────────────────────────────────────────────────────────────
# FUNCTION 7 — Silence Detection
# ─────────────────────────────────────────────────────────────────────
def detect_silences(transcript: list, threshold: float = 2.0) -> list:
    """
    Detect gaps between segments longer than threshold seconds.
    These represent pauses in the conversation.
    """
    silences = []
    for i in range(1, len(transcript)):
        gap = transcript[i]["start_time"] - transcript[i - 1]["end_time"]
        if gap > threshold:
            silences.append({
                "from":     transcript[i - 1]["end_time"],
                "to":       transcript[i]["start_time"],
                "duration": round(gap, 2)
            })
    return silences


# ─────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT — run_phase1
# ─────────────────────────────────────────────────────────────────────
def run_phase1(
    audio_path:  str,
    hf_token:    str,
    model_size:  str   = "small",
    silence_threshold: float = 2.0,
    output_json: str   = None,
    output_txt:  str   = None
) -> dict:
    """
    Full Phase 1 pipeline.

    Args:
        audio_path:         path to WAV or MP3 file
        hf_token:           HuggingFace token
        model_size:         Whisper model size (small recommended)
        silence_threshold:  minimum gap in seconds to flag as silence
        output_json:        optional path to save JSON output
        output_txt:         optional path to save readable TXT output

    Returns:
        dict with keys:
            audio_file, duration_seconds, language,
            transcript, silences, summary
    """
    # Step 1 — Preprocess
    wav_path   = convert_to_wav(audio_path)
    clean_path = denoise_audio(wav_path)
    duration   = librosa.get_duration(path=clean_path)

    # Step 2 — Transcribe
    whisper_segs, language = run_transcription(clean_path, model_size)

    # Step 3 — Diarize
    diar_segs = run_diarization(clean_path, hf_token)

    # Step 4 — Merge + Label
    transcript = merge_transcript(whisper_segs, diar_segs)
    transcript = label_speakers(transcript)
    silences   = detect_silences(transcript, silence_threshold)

    # Step 5 — Build output
    result = {
        "audio_file":       audio_path,
        "duration_seconds": round(duration, 2),
        "language":         language,
        "transcript":       transcript,
        "silences":         silences,
        "summary": {
            "total_segments":    len(transcript),
            "agent_segments":    len([t for t in transcript if t["speaker"] == "Agent"]),
            "customer_segments": len([t for t in transcript if t["speaker"] == "Customer"]),
            "long_silences":     len(silences),
            "total_turns":       len(transcript)
        }
    }

    # Step 6 — Save if paths provided
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    if output_txt:
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(f"Call Duration: {duration:.2f}s | Language: {language}\n")
            f.write("=" * 60 + "\n\n")
            for t in transcript:
                f.write(f"[{t['start_time']:.2f}s → {t['end_time']:.2f}s] "
                        f"{t['speaker']}: {t['text']}\n")
            f.write("\n--- Long Silences ---\n")
            for s in silences:
                f.write(f"  {s['from']}s → {s['to']}s ({s['duration']}s gap)\n")

    return result