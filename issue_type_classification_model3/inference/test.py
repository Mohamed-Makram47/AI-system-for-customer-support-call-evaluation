import argparse
import json
from typing import Any, Dict, Iterable, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel, AutoTokenizer, RobertaTokenizerFast


REPO_ID = "Mohamed-Makram77/issue-type-classification-robertaLarge"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DualHeadClassifier(nn.Module):
    """
    RoBERTa encoder + coarse head + fine head.

    This matches your notebook architecture:
      pooled CLS -> coarse_head
      pooled CLS + all coarse probabilities -> fine_head
    """

    def __init__(self, model_name: str, num_fine: int, num_coarse: int, dropout: float = 0.1):
        super().__init__()

        # Load only the config first.
        # The trained encoder weights come from best_roberta.pt.
        encoder_config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_config(encoder_config)

        hidden = self.encoder.config.hidden_size

        self.coarse_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_coarse),
        )

        self.fine_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + num_coarse, num_fine),
        )

        self.num_coarse = num_coarse

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]

        coarse_logits = self.coarse_head(pooled)
        coarse_probs = F.softmax(coarse_logits, dim=-1)

        fine_input = torch.cat([pooled, coarse_probs], dim=-1)
        fine_logits = self.fine_head(fine_input)

        return coarse_logits, fine_logits


def load_json_from_hf(repo_id: str, filename: str) -> Any:
    path = hf_hub_download(repo_id=repo_id, filename=filename)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_id_map(raw: Union[Dict[str, str], List[str]]) -> Dict[int, str]:
    if isinstance(raw, list):
        return {i: label for i, label in enumerate(raw)}

    return {int(k): v for k, v in raw.items()}


def normalize_coarse_to_fine(raw: Dict[str, Iterable[Any]]) -> Dict[int, List[int]]:
    return {int(k): [int(x) for x in v] for k, v in raw.items()}


def load_tokenizer(repo_id: str, model_name: str):
    """
    Your tokenizer files are uploaded at the repo root:
      tokenizer_config.json
      tokenizer.json

    This function also supports a tokenizer/ subfolder as a fallback.
    """

    attempts = [
        ("AutoTokenizer from repo root", lambda: AutoTokenizer.from_pretrained(repo_id)),
        ("RobertaTokenizerFast from repo root", lambda: RobertaTokenizerFast.from_pretrained(repo_id)),
        ("AutoTokenizer from tokenizer/ subfolder", lambda: AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer")),
        ("RobertaTokenizerFast from tokenizer/ subfolder", lambda: RobertaTokenizerFast.from_pretrained(repo_id, subfolder="tokenizer")),
        ("AutoTokenizer from base model", lambda: AutoTokenizer.from_pretrained(model_name)),
    ]

    last_error = None

    for name, loader in attempts:
        try:
            tokenizer = loader()
            print(f"Tokenizer loaded: {name}")
            return tokenizer
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not load tokenizer. Last error: {last_error}") from last_error


def torch_load_checkpoint(path: str) -> Any:
    """
    Loads your PyTorch checkpoint.
    """

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

        # Your notebook saved m_save.state_dict(), so this should normally run.
        if all(isinstance(k, str) for k in checkpoint.keys()):
            return checkpoint

    raise TypeError(
        "The checkpoint is not a state_dict. "
        "Expected best_roberta.pt to contain model.state_dict()."
    )


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = state_dict

    for prefix in ["module.", "_orig_mod."]:
        if any(k.startswith(prefix) for k in cleaned.keys()):
            cleaned = {
                (k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in cleaned.items()
            }

    return cleaned


def download_weights(repo_id: str) -> str:
    """
    Your repo has best_roberta.pt at the root.
    This also supports model/best_roberta.pt just in case.
    """

    try:
        return hf_hub_download(repo_id=repo_id, filename="best_roberta.pt")
    except Exception:
        return hf_hub_download(repo_id=repo_id, filename="model/best_roberta.pt")


class IssueTypeClassifier:
    def __init__(self, repo_id: str = REPO_ID, device: torch.device = DEVICE):
        self.repo_id = repo_id
        self.device = device

        print(f"Using repo: {repo_id}")
        print(f"Using device: {device}")

        self.cfg = load_json_from_hf(repo_id, "training_config.json")
        self.id2fine = normalize_id_map(load_json_from_hf(repo_id, "id2fine.json"))
        self.id2coarse = normalize_id_map(load_json_from_hf(repo_id, "id2coarse.json"))
        self.coarse_to_fine_ids = normalize_coarse_to_fine(
            load_json_from_hf(repo_id, "coarse_to_fine_ids.json")
        )

        self.model_name = self.cfg.get("model_name", "roberta-large")
        self.max_len = int(self.cfg.get("max_len", 128))
        self.num_fine = int(self.cfg.get("num_fine_classes", len(self.id2fine)))
        self.num_coarse = int(self.cfg.get("num_coarse_classes", len(self.id2coarse)))

        print(f"Base model: {self.model_name}")
        print(f"Max length: {self.max_len}")
        print(f"Fine classes: {self.num_fine}")
        print(f"Coarse classes: {self.num_coarse}")

        self.tokenizer = load_tokenizer(repo_id, self.model_name)

        self.model = DualHeadClassifier(
            model_name=self.model_name,
            num_fine=self.num_fine,
            num_coarse=self.num_coarse,
            dropout=0.1,
        )

        weights_path = download_weights(repo_id)
        checkpoint = torch_load_checkpoint(weights_path)
        state_dict = clean_state_dict_keys(extract_state_dict(checkpoint))

        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print("\nCheckpoint key examples:")
            for key in list(state_dict.keys())[:20]:
                print("  ", key)

            raise RuntimeError(
                "The checkpoint does not match the DualHeadClassifier architecture. "
                "Check that best_roberta.pt was saved from the same notebook architecture."
            ) from e

        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully.\n")

    def _predict_batch(self, texts: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        enc = self.tokenizer(
            texts,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            coarse_logits, fine_logits = self.model(input_ids, attention_mask)

            coarse_probs = F.softmax(coarse_logits, dim=-1)
            coarse_conf, coarse_pred_ids = coarse_probs.max(dim=-1)

            # Hard hierarchical decoding:
            # keep only fine labels that belong to the predicted coarse class.
            masked_fine_logits = fine_logits.clone()

            for i, coarse_id in enumerate(coarse_pred_ids.tolist()):
                valid_fine_ids = self.coarse_to_fine_ids[int(coarse_id)]

                mask_vec = torch.full(
                    (fine_logits.size(1),),
                    float("-inf"),
                    device=self.device,
                )

                mask_vec[valid_fine_ids] = 0.0
                masked_fine_logits[i] = masked_fine_logits[i] + mask_vec

            fine_probs = F.softmax(masked_fine_logits, dim=-1)
            fine_conf, fine_pred_ids = fine_probs.max(dim=-1)

            top_values, top_indices = torch.topk(
                fine_probs,
                k=min(top_k, fine_probs.size(1)),
                dim=-1,
            )

        results = []

        for i, text in enumerate(texts):
            coarse_id = int(coarse_pred_ids[i].cpu().item())
            fine_id = int(fine_pred_ids[i].cpu().item())

            top_predictions = []

            for score, idx in zip(top_values[i].cpu().tolist(), top_indices[i].cpu().tolist()):
                top_predictions.append(
                    {
                        "fine_id": int(idx),
                        "fine_label": self.id2fine[int(idx)],
                        "confidence": float(score),
                    }
                )

            results.append(
                {
                    "text": text,
                    "coarse_id": coarse_id,
                    "coarse_label": self.id2coarse[coarse_id],
                    "coarse_confidence": float(coarse_conf[i].cpu().item()),
                    "fine_id": fine_id,
                    "fine_label": self.id2fine[fine_id],
                    "fine_confidence": float(fine_conf[i].cpu().item()),
                    "top_fine_predictions": top_predictions,
                }
            )

        return results

    def predict(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 16,
        top_k: int = 5,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        single_input = isinstance(texts, str)

        if single_input:
            text_list = [texts]
        else:
            text_list = list(texts)

        all_results = []

        for start in range(0, len(text_list), batch_size):
            batch_texts = text_list[start:start + batch_size]
            all_results.extend(self._predict_batch(batch_texts, top_k=top_k))

        return all_results[0] if single_input else all_results


def print_prediction(result: Dict[str, Any]) -> None:
    print("=" * 100)
    print(f"Text              : {result['text']}")
    print(f"Predicted coarse  : {result['coarse_label']}  ({result['coarse_confidence']:.4f})")
    print(f"Predicted fine    : {result['fine_label']}  ({result['fine_confidence']:.4f})")
    print("Top fine labels:")

    for item in result["top_fine_predictions"]:
        print(f"  - {item['fine_label']:<35} {item['confidence']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="RoBERTa-only hierarchical issue-type classifier")

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text to classify. If omitted, demo examples are used.",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        default=REPO_ID,
        help="Hugging Face repo ID.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top fine labels to print.",
    )

    args = parser.parse_args()

    classifier = IssueTypeClassifier(repo_id=args.repo_id)

    if args.text is not None:
        texts = [args.text]
    else:
        texts = [
            "My card was swallowed by the ATM, what should I do?",
            "I was charged twice for the same transaction.",
            "How long does it take for a new card to arrive?",
            "I forgot my passcode and now I cannot log in.",
            "Can I use Apple Pay with my account?",
            "Why was my top-up rejected?",
            "I need to change my personal details.",
        ]

    predictions = classifier.predict(
        texts,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )

    if isinstance(predictions, dict):
        print_prediction(predictions)
    else:
        for prediction in predictions:
            print_prediction(prediction)


if __name__ == "__main__":
    main()