import torch
import torch.nn.functional as F
import requests
from transformers import AutoTokenizer
from model import DualHeadClassifier 

MODEL_ID = "Mohamed-Makram77/hierarchical-issue_classification_model"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT_INPUT = "My card was declined at the grocery store today and I don't know why."

# Load Mappings
fine_map = requests.get(f"https://huggingface.co/{MODEL_ID}/resolve/main/id2fine.json").json()
coarse_map = requests.get(f"https://huggingface.co/{MODEL_ID}/resolve/main/id2coarse.json").json()

# Load Model
model = DualHeadClassifier(num_coarse=10, num_fine=78)
weights_url = f"https://huggingface.co/{MODEL_ID}/resolve/main/dual_head_model.pt"
state_dict = torch.hub.load_state_dict_from_url(weights_url, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE).eval()

# Run
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
inputs = tokenizer(TEXT_INPUT, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)

with torch.no_grad():
    c_logits, f_logits = model(**inputs)
    f_id = str(torch.argmax(f_logits, dim=-1).item())
    c_id = str(torch.argmax(c_logits, dim=-1).item())

print(f"\nResult: {fine_map[f_id]} (Category: {coarse_map[c_id]})")