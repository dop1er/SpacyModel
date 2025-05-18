# src/evaluate.py

import os
import argparse
import torch
import numpy as np
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from seqeval.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer
from src.model import MultiTaskModel
from src.data_utils import tokenize_and_align_labels, load_labeled

def evaluate_pos(model, tokenizer, labeled_path, label_map, batch_size=32, device="cpu"):
    examples = load_labeled(labeled_path)
    pos2id = {label: idx for label, idx in label_map.items()}
    id2pos = {idx: label for label, idx in label_map.items()}

    tokenized = tokenize_and_align_labels(examples, tokenizer, pos2id, {}, max_length=512)
    dataset = Dataset.from_dict(tokenized)
    loader = DataLoader(dataset, batch_size=batch_size)

    model.to(device).eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["pos_labels"].to(device)
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1)

            for p_seq, t_seq, m in zip(preds.cpu().numpy(), labels.cpu().numpy(), attention_mask.cpu().numpy()):
                pred_labels, true_labels = [], []
                for p, t, mask in zip(p_seq, t_seq, m):
                    if mask and t != -100:
                        pred_labels.append(id2pos[p])
                        true_labels.append(id2pos[t])
                all_preds.append(pred_labels)
                all_labels.append(true_labels)

    print("POS Report:\n", classification_report(all_labels, all_preds))
    print("Accuracy:", accuracy_score(all_labels, all_preds))

def evaluate_sts(model, tokenizer, batch_size=32, device="cpu"):
    ds = load_dataset("glue", "stsb", split="validation")
    enc = tokenizer(ds["sentence1"], ds["sentence2"], truncation=True, padding="max_length", return_tensors="pt")
    input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
    labels = np.array(ds["label"], dtype=float) / 5.0

    loader = DataLoader(list(zip(input_ids, attention_mask)), batch_size=batch_size)
    model.to(device).eval()
    cos = torch.nn.CosineSimilarity(dim=1)
    vecs = []
    with torch.no_grad():
        for ids, mask in loader:
            ids, mask = ids.to(device), mask.to(device)
            _, out = model(input_ids=ids, attention_mask=mask)
            vecs.append(out)
    vecs = torch.cat(vecs)
    sims = cos(vecs[0::2], vecs[1::2]).cpu().numpy()
    print("STS-B Pearson:", pearsonr(sims, labels)[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", choices=["pos", "sts", "all"], default="all")
    parser.add_argument("--labeled_data", type=str, default="data/raw/ru_syntagrus-ud-2.16")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--label_map", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = MultiTaskModel.from_pretrained(args.checkpoint)
    with open(args.label_map, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    if args.task in ("pos", "all"):
        evaluate_pos(model, tokenizer, args.labeled_data, label_map, args.batch_size, args.device)
    if args.task in ("sts", "all"):
        evaluate_sts(model, tokenizer, args.batch_size, args.device)
