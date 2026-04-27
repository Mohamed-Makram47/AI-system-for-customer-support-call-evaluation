"""
src/classifier.py — BERT (RoBERTa) dual-head issue-type classifier.

Classifies a transcript utterance into one of 78 fine-grained issue classes
(and 10 coarse categories).  Used by the updated RAG pipeline to scope FAISS
retrieval to the correct class-specific compliance manual.

Architecture
────────────
  RoBERTa encoder
       ├─ coarse head  → 10 classes  (category level)
       └─ fine head    → 78 classes  (issue level, boosted by coarse probs)

Loading order:
  1. If a local `dual_head_model.pt` exists in MODEL_DIR  → load from disk.
  2. Otherwise                                            → download from HuggingFace Hub.

Usage:
  from src.classifier import load_classifier, classify

  model, tokenizer, id2fine, id2coarse = load_classifier()
  result = classify("My card was declined at the store.", model, tokenizer, id2fine, id2coarse)
  print(result)
  # {'coarse_id': 2, 'coarse_label': 'card_functionality',
  #  'fine_id': 26, 'fine_label': 'declined_card_payment'}
"""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

# HuggingFace model repo (used as fallback when no local .pt file is found)
_HF_MODEL_ID = "Mohamed-Makram77/hierarchical-issue_classification_model"

# Default local directory that holds id2fine.json, id2coarse.json, tokenizer.json, etc.
_DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "model",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model Definition ─────────────────────────────────────────────────────────

class DualHeadClassifier(nn.Module):
    """Dual-head classifier: coarse (10 classes) + fine (78 classes)."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_coarse: int = 10,
        num_fine: int = 78,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        # Coarse head (10 classes)
        self.coarse_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_coarse),
        )

        # Fine head (78 classes)
        self.fine_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_fine),
        )

        # Soft routing: coarse probabilities inform the fine prediction
        self.coarse_to_fine = nn.Linear(num_coarse, num_fine, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]

        coarse_logits = self.coarse_head(cls)
        coarse_probs = F.softmax(coarse_logits, dim=-1)

        # Add coarse information to fine prediction
        fine_logits = self.fine_head(cls) + self.coarse_to_fine(coarse_probs)

        return coarse_logits, fine_logits


# ── Loader ────────────────────────────────────────────────────────────────────

def load_classifier(model_dir: str | None = None):
    """
    Load the dual-head classifier, tokenizer, and label maps.

    Parameters
    ----------
    model_dir : str, optional
        Path to the local model directory. Defaults to
        ``model/`` at the project root.

    Returns
    -------
    model      : DualHeadClassifier  (on DEVICE, eval mode)
    tokenizer  : AutoTokenizer
    id2fine    : dict[str, str]       mapping "0"→"Refund_not_showing_up", …
    id2coarse  : dict[str, str]       mapping "0"→"account_and_balance", …
    """
    model_dir = model_dir or _DEFAULT_MODEL_DIR
    model_path = Path(model_dir)

    # ── label maps ────────────────────────────────────────────────────────
    with open(model_path / "id2fine.json", encoding="utf-8") as f:
        id2fine: dict[str, str] = json.load(f)
    with open(model_path / "id2coarse.json", encoding="utf-8") as f:
        id2coarse: dict[str, str] = json.load(f)

    # ── tokenizer ─────────────────────────────────────────────────────────
    # Check for tokenizer/ subdirectory first, then fall back to model_dir
    # itself (works when tokenizer.json sits at the root of model_dir),
    # and finally fall back to HuggingFace.
    tokenizer_dir = model_path / "tokenizer"
    if tokenizer_dir.exists() and tokenizer_dir.is_dir():
        print(f"[classifier] Loading tokenizer from local dir: {tokenizer_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    elif (model_path / "tokenizer.json").exists():
        print(f"[classifier] Loading tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    else:
        print(f"[classifier] Loading tokenizer from HuggingFace: {_HF_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(_HF_MODEL_ID)

    # ── model weights ─────────────────────────────────────────────────────
    model = DualHeadClassifier(num_coarse=10, num_fine=78)

    local_pt = model_path / "dual_head_model.pt"
    if local_pt.exists():
        print(f"[classifier] Loading weights from local: {local_pt}")
        state_dict = torch.load(str(local_pt), map_location=DEVICE)
    else:
        weights_url = (
            f"https://huggingface.co/{_HF_MODEL_ID}/resolve/main/dual_head_model.pt"
        )
        print(f"[classifier] Downloading weights from: {weights_url}")
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url, map_location=DEVICE
        )

    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    print(f"[classifier] Model loaded on {DEVICE}  (OK)")

    return model, tokenizer, id2fine, id2coarse


# ── Inference ─────────────────────────────────────────────────────────────────

def classify(
    text: str,
    model: DualHeadClassifier,
    tokenizer,
    id2fine: dict[str, str],
    id2coarse: dict[str, str],
) -> dict:
    """
    Classify a single utterance and return coarse + fine predictions.

    Returns
    -------
    dict with keys:
        coarse_id, coarse_label, fine_id, fine_label
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        coarse_logits, fine_logits = model(**inputs)

    coarse_id = coarse_logits.argmax(dim=-1).item()
    fine_id = fine_logits.argmax(dim=-1).item()

    return {
        "coarse_id": coarse_id,
        "coarse_label": id2coarse[str(coarse_id)],
        "fine_id": fine_id,
        "fine_label": id2fine[str(fine_id)],
    }


# ── Convenience wrapper ──────────────────────────────────────────────────────

class ClassifierPipeline:
    """
    Stateful wrapper — load once, classify many.

    Usage:
        pipe = ClassifierPipeline()
        result = pipe("I lost my card and need a replacement.")
    """

    def __init__(self, model_dir: str | None = None):
        self.model, self.tokenizer, self.id2fine, self.id2coarse = load_classifier(
            model_dir
        )

    def __call__(self, text: str) -> dict:
        return classify(text, self.model, self.tokenizer, self.id2fine, self.id2coarse)


# ── CLI quick-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipe = ClassifierPipeline()

    test_texts = [
        "My card was declined at the grocery store today and I don't know why.",
        "I need to change my PIN number please.",
        "Why was I charged a fee for the ATM withdrawal?",
        "I want to cancel my account.",
        "How do I top up my balance using a bank transfer?",
    ]

    for text in test_texts:
        result = pipe(text)
        print(
            f"  [{result['coarse_label']:>28}]  "
            f"{result['fine_label']:<40}  <-  \"{text[:60]}...\""
        )
