"""
Baseline B1: Standard fine-tuning of Qwen2.5 for task only.
Loss: L_task = CrossEntropyLoss. No bias suppression, no adversarial head, no projection.
Saves checkpoint and extracts final hidden states for probing.
"""
import os
import torch
import json
from pathlib import Path
from typing import Optional
from datasets import Dataset
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CHECKPOINT_DIR, REPRESENTATIONS_DIR, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR, ensure_dirs, get_device
from data.loaders import get_qwen_tokenizer
from models.qwen_task import QwenTaskModel


def _collate_batch(batch):
    """Collate batch with input_ids, attention_mask, label, sensitive_attribute (Bias in Bios or any labeled dataset)."""
    input_ids = torch.stack([torch.tensor(b["input_ids"]) for b in batch])
    attention_mask = torch.stack([torch.tensor(b["attention_mask"]) for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    sensitive = [b["sensitive_attribute"] for b in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "sensitive_attribute": sensitive,
    }


def run_b1(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    num_labels: int = 2,
    output_dir: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    device: Optional[str] = None,
    save_representations: bool = True,
) -> tuple:
    """
    Train B1 (task-only), save checkpoint, extract and save hidden states.
    Returns (model, checkpoint_path, representations_path).
    """
    ensure_dirs()
    output_dir = output_dir or os.path.join(CHECKPOINT_DIR, "b1_standard")
    os.makedirs(output_dir, exist_ok=True)
    device = device or get_device()

    model = QwenTaskModel(model_name=model_name, num_labels=num_labels)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    pin_memory = device.startswith("cuda") if isinstance(device, str) else False
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
        pin_memory=pin_memory,
    )
    eval_loader = None
    if eval_dataset and len(eval_dataset) > 0:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            collate_fn=_collate_batch,
            pin_memory=pin_memory,
        )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            out = model(**inputs)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"B1 Epoch {epoch + 1}/{epochs} loss: {total_loss / len(train_loader):.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(output_dir, "pytorch_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"model_name": model_name, "num_labels": num_labels},
    }, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # Extract final hidden states for probing (use train + eval)
    reps_path = None
    if save_representations:
        reps_dir = os.path.join(REPRESENTATIONS_DIR, "b1_standard")
        os.makedirs(reps_dir, exist_ok=True)
        model.eval()
        all_hidden = []
        all_labels = []
        all_sensitive = []
        full_dataset = train_dataset
        if eval_dataset and len(eval_dataset) > 0:
            from datasets import concatenate_datasets
            full_dataset = concatenate_datasets([train_dataset, eval_dataset])
        loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            collate_fn=_collate_batch,
        )
        with torch.no_grad():
            for batch in loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    return_hidden=True,
                )
                all_hidden.append(out["hidden_states"].cpu())
                all_labels.append(batch["labels"])
                all_sensitive.extend(batch["sensitive_attribute"])
        hidden_t = torch.cat(all_hidden, dim=0)
        labels_t = torch.cat(all_labels, dim=0)
        reps_path = os.path.join(reps_dir, "hidden_and_metadata.pt")
        torch.save({
            "hidden_states": hidden_t,
            "labels": labels_t,
            "sensitive_attributes": all_sensitive,
        }, reps_path)
        meta_path = os.path.join(reps_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "n_samples": hidden_t.size(0),
                "hidden_dim": hidden_t.size(1),
                "checkpoint": ckpt_path,
            }, f, indent=2)
        print(f"Representations saved to {reps_path}")

    return model, ckpt_path, reps_path
