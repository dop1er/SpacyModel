# src/train.py

import os
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, default_data_collator
from datasets import Dataset
from tqdm import tqdm

from src.model import MultiTaskModel, MultiTaskConfig
from src.data_utils import load_labeled, tokenize_and_align_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Train MultiTaskModel for POS tagging and projection head")
    parser.add_argument("--labeled_data", type=str, required=True,
                        help="Path to labeled data (.conllu file or directory)")
    parser.add_argument("--pos_label_map", type=str, required=True,
                        help="JSON file with mapping POS-tag->id")
    parser.add_argument("--morph_label_map", type=str, required=True,
                        help="JSON file with mapping morph-feat->id")
    parser.add_argument("--encoder_name", type=str, default="DeepPavlov/rubert-base-cased",
                        help="Pretrained encoder name or path")
    parser.add_argument("--output_dir", type=str, default="checkpoints/multitask_model",
                        help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu or cuda), default auto-detect")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label mappings
    with open(args.pos_label_map, 'r', encoding='utf-8') as f:
        pos_label2id = json.load(f)
    with open(args.morph_label_map, 'r', encoding='utf-8') as f:
        morph_label2id = json.load(f)

    # Initialize model and config
    config = MultiTaskConfig(
        encoder_name_or_path=args.encoder_name,
        num_pos_labels=len(pos_label2id),
        proj_size=None
    )
    model = MultiTaskModel(config).to(device)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)

    # Load and preprocess labeled data
    print("Loading labeled data...")
    raw_examples = load_labeled(args.labeled_data)
    tokenized = tokenize_and_align_labels(
        raw_examples,
        tokenizer,
        pos_label2id=pos_label2id,
        morph_label2id=morph_label2id,
        max_length=args.max_length
    )
    dataset = Dataset.from_dict(tokenized)

    # DataLoader with collate_fn to convert lists to tensors
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=default_data_collator
    )

    # Optimizer, scheduler, and loss function
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Training loop
    print(f"Starting training on {device} for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pos_labels = batch['pos_labels'].to(device)

            optimizer.zero_grad()
            pos_logits, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            loss_pos = loss_fn(
                pos_logits.view(-1, pos_logits.size(-1)),
                pos_labels.view(-1)
            )
            loss_pos.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss_pos.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{args.epochs} done. Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        output_dir = Path(args.output_dir) / f"epoch-{epoch}"
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    print("Training completed.")


if __name__ == '__main__':
    main()
