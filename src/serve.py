# src/serve.py

import os
import json
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from transformers import AutoTokenizer
from src.model import MultiTaskModel

# Paths (can override via ENV)
CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "checkpoints/multitask_model/epoch-3")
LABEL_MAP_PATH = os.getenv("POS_LABEL_MAP", "configs/pos_label_map.json")

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = MultiTaskModel.from_pretrained(CHECKPOINT)
model.to(device).eval()

# Load id->label map
with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
    label_map = json.load(f)
id2pos = {v: k for k, v in label_map.items()}


class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    tokens: List[str]
    pos_tags: List[str]
    embeddings: List[float]


app = FastAPI()

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    # Tokenize
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    # Predict
    pos_logits, projection = model(**inputs)
    pred_ids = torch.argmax(pos_logits, dim=-1).squeeze(0).cpu().tolist()

    # Convert tokens & tags
    all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0).cpu().tolist())
    tokens, pos_tags = [], []
    for tok, pid in zip(all_tokens, pred_ids):
        if tok in tokenizer.all_special_tokens:
            continue
        tokens.append(tok)
        pos_tags.append(id2pos.get(pid, "X"))

    embeddings = projection.squeeze(0).detach().cpu().tolist()
    return AnalyzeResponse(tokens=tokens, pos_tags=pos_tags, embeddings=embeddings)
