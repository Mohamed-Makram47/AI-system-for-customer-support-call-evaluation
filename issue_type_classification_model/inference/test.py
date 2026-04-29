import json
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
from huggingface_hub import hf_hub_download

REPO_ID = "Mohamed-Makram77/banking-issue-classifier"

# ── Load model and tokenizer ──────────────────────────────────────────────────
print("Loading model from Hugging Face...")
tokenizer = BertTokenizerFast.from_pretrained(REPO_ID)
model     = BertForSequenceClassification.from_pretrained(REPO_ID)
model.eval()
print("Model loaded successfully!")
print()

# ── Load label mapping ────────────────────────────────────────────────────────
id2label_path = hf_hub_download(repo_id=REPO_ID, filename="id2label.json")
with open(id2label_path) as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

print(f"Classes loaded: {len(id2label)}")
print()

# ── Test sentences ────────────────────────────────────────────────────────────
test_sentences = [
    "I was charged twice this month and I want a refund.",
    "My card has not arrived yet, it has been two weeks.",
    "The ATM declined my card even though I have money.",
    "I forgot my PIN and now my card is blocked.",
    "Thank you for your patience, let me check your account.",
    "Hi I am very happy with your service, keep up the good work!",
    "I was charged twice this month and I want a refund.",
    "My card has not arrived yet, it has been two weeks.",
    "The ATM declined my card even though I have money.",
    "I forgot my PIN and now my card is blocked.",
    "Thank you for your patience, let me check your account.",
    "Hi I am very happy with your service, keep up the good work!",
]

print("Testing individual sentences:")
print("=" * 65)

for sentence in test_sentences:
    inputs = tokenizer(
        sentence,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=-1)
        conf, idx = probs.max(dim=-1)

    label = id2label[idx.item()]
    print(f"Sentence  : {sentence}")
    print(f"Predicted : {label}")
    print(f"Confidence: {conf.item()*100:.1f}%")
    print()