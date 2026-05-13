"""
src/topic_segmenter.py — Detects topic shifts in a transcript using embedding similarity.

Algorithm:
  1. Filter customer turns with >= min_words words  -> meaningful_turns
  2. Embed them with normalize_embeddings=True
  3. Compute cosine similarity between consecutive embeddings
     (dot product of unit vectors == cosine similarity)
  4. Flag a shift at position i if sim[i] < threshold AND sim[i+1] < threshold
  5. Boundary timestamp = start_time of the turn immediately after the shift
  6. Split ALL turns at those boundaries into segments
"""
import numpy as np


def segment_transcript(
    turns: list[dict],
    embedder,
    min_words: int = 5,
    threshold: float = 0.40,
) -> list[dict]:
    """
    Detect topic shifts and return a list of segment dicts.

    Parameters
    ----------
    turns     : full transcript -- dicts with speaker, text, start_time
    embedder  : SentenceTransformer instance (already loaded)
    min_words : minimum word count for a customer turn to be considered
    threshold : cosine-similarity threshold below which a turn is "different"

    Returns
    -------
    List of segment dicts, each containing:
        segment_id, start_time, end_time, turns, customer_turns, agent_turns
    """
    # -- Step 1: filter meaningful customer turns -----------------------------
    meaningful_turns = [
        t for t in turns
        if t["speaker"] == "Customer" and len(t["text"].split()) >= min_words
    ]

    # Need at least 3 meaningful turns to detect a double-dip shift
    if len(meaningful_turns) < 3:
        return _build_segments(turns, boundary_times=[], all_end=_last_time(turns))

    # -- Step 2: embed --------------------------------------------------------
    texts      = [t["text"] for t in meaningful_turns]
    embeddings = embedder.encode(texts, normalize_embeddings=True)  # (N, D) float32

    # -- Step 3: cosine similarities between consecutive embeddings -----------
    # Since vectors are unit-normalised, dot product == cosine similarity
    sims = np.array([
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ])

    # -- Step 4: flag shifts -- consecutive pair both below threshold ---------
    boundary_times: list[float] = []
    i = 0
    while i < len(sims) - 1:
        if sims[i] < threshold and sims[i + 1] < threshold:
            # -- Step 5: boundary = start_time of the turn after the shift ----
            shift_turn_idx = i + 1          # index into meaningful_turns
            boundary_times.append(meaningful_turns[shift_turn_idx]["start_time"])
            i += 2                          # skip next sim (already consumed)
        else:
            i += 1

    # -- Step 6 & 7: split all turns at boundaries and build segment dicts ----
    return _build_segments(turns, boundary_times, all_end=_last_time(turns))


# -- helpers ------------------------------------------------------------------

def _last_time(turns: list[dict]) -> float:
    """Return the latest end_time (or start_time) seen across all turns."""
    times = [t.get("end_time", t["start_time"]) for t in turns]
    return max(times) if times else 0.0


def _build_segments(
    turns: list[dict],
    boundary_times: list[float],
    all_end: float,
) -> list[dict]:
    """
    Slice `turns` at each boundary timestamp and package into segment dicts.
    """
    boundaries = sorted(set(boundary_times))
    starts = [0.0] + boundaries
    ends   = boundaries + [all_end]

    segments: list[dict] = []

    for seg_id, (seg_start, seg_end) in enumerate(zip(starts, ends), start=1):
        if seg_id == len(starts):
            # Last segment: inclusive of everything from seg_start onward
            seg_turns = [t for t in turns if t["start_time"] >= seg_start]
        else:
            seg_turns = [
                t for t in turns
                if seg_start <= t["start_time"] < seg_end
            ]

        customer_turns = [t["text"] for t in seg_turns if t["speaker"] == "Customer"]
        agent_turns    = [t["text"] for t in seg_turns if t["speaker"] == "Agent"]

        segments.append({
            "segment_id":     seg_id,
            "start_time":     seg_start,
            "end_time":       seg_end,
            "turns":          seg_turns,
            "customer_turns": customer_turns,
            "agent_turns":    agent_turns,
        })

    return segments


def print_segments(segments: list[dict]) -> None:
    """Print a human-readable summary of each detected segment."""
    print(f"\n  {'Seg':>4}  {'Time range':>20}  {'Turns':>6}  First customer utterance")
    print("  " + "-" * 80)
    for seg in segments:
        time_range = f"{seg['start_time']:.1f}s - {seg['end_time']:.1f}s"
        n_turns    = len(seg["turns"])
        first_cust = seg["customer_turns"][0][:70] if seg["customer_turns"] else "(none)"
        print(f"  {seg['segment_id']:>4}  {time_range:>20}  {n_turns:>6}  \"{first_cust}\"")
    print()
