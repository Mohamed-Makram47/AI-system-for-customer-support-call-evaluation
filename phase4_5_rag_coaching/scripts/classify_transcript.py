"""
Reusable majority-vote classification for a single transcript.
"""

def classify(utterances: list[dict], classifier) -> str:
    """
    Classifies a transcript using majority vote across customer utterances.

    Args:
        utterances (list[dict]): A list of all dialogue turns (speaker, text).
        classifier: The ClassifierPipeline instance.

    Returns:
        str: The predicted fine label string.
    """
    label_counts = {}
    label_scores = {}
    for u in utterances:
        if u["speaker"] == "customer":
            prediction = classifier(u["text"])
            lbl        = prediction["fine_label"]
            score      = prediction.get("confidence", 0.0)
            if lbl == "neutral":
                continue  # skip neutral votes — filler turns should not influence intent
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            label_scores[lbl] = label_scores.get(lbl, 0.0) + score

    if not label_counts:
        return "unknown"

    predicted_label = max(
        label_counts.keys(),
        key=lambda k: (label_counts[k], label_scores[k])
    )
    return predicted_label
