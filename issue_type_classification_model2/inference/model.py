import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class DualHeadClassifier(nn.Module):
    def __init__(self, model_name='roberta-base', num_coarse=10, num_fine=78, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        
        # Coarse head (10 classes)
        self.coarse_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_coarse)
        )
        
        # Fine head (78 classes)
        self.fine_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_fine)
        )
        
        # Soft routing layer
        self.coarse_to_fine = nn.Linear(num_coarse, num_fine, bias=False)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        
        coarse_logits = self.coarse_head(cls)
        coarse_probs = F.softmax(coarse_logits, dim=-1)
        
        # Add coarse information to fine prediction
        fine_logits = self.fine_head(cls) + self.coarse_to_fine(coarse_probs)
        
        return coarse_logits, fine_logits