"""
Baseline B2: Adversarial Debiasing (static).
Shared Qwen backbone, task classifier head, bias classifier head, GRL.
L_total = L_task + lambda * L_bias. No stability regularization, no simulated adaptation.

Training modes:
- use_lora=False (default): AdamW on **all** parameters (backbone + heads). GRL gradients update the backbone.
- use_lora=True: **Base backbone frozen**; train **LoRA adapters + task + bias heads** (recommended when you
  want controlled representation change without full fine-tune; matches “bias lives in the backbone” story).
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
from config import (
    CHECKPOINT_DIR,
    REPRESENTATIONS_DIR,
    DEFAULT_ADV_BIAS_LOSS_BALANCE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_LAMBDA_BIAS,
    ensure_dirs,
    get_device,
)
from baselines.b1_standard import _collate_batch
from models.adversarial import QwenAdversarialModel, bias_loss_term


def apply_lora_to_adversarial_backbone(
    model: QwenAdversarialModel,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> QwenAdversarialModel:
    """Wrap backbone with PEFT LoRA; base weights frozen, only LoRA (+ heads) trainable."""
    from peft import LoraConfig, TaskType, get_peft_model

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model.backbone = get_peft_model(model.backbone, peft_config)
    return model


def load_b2_from_checkpoint(
    ckpt_path: str,
    model_name: str,
    device: str,
    num_task_labels: int,
    num_bias_labels: int = 2,
) -> QwenAdversarialModel:
    """Load B2; if checkpoint was trained with LoRA, rebuild adapter wrapper before load_state_dict."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("config", {})
    use_lora = bool(cfg.get("use_lora", False))
    model = QwenAdversarialModel(
        model_name=model_name,
        num_task_labels=num_task_labels,
        num_bias_labels=num_bias_labels,
    )
    if use_lora:
        apply_lora_to_adversarial_backbone(
            model,
            lora_r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        )
    model = model.to(device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    return model


def run_b2(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    num_labels: int = 2,
    num_bias_labels: int = 2,
    lambda_bias: float = DEFAULT_LAMBDA_BIAS,
    output_dir: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    device: Optional[str] = None,
    save_representations: bool = True,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    balance_bias_loss: bool = DEFAULT_ADV_BIAS_LOSS_BALANCE,
) -> tuple:
    """
    Train B2 adversarial debiasing. Returns (model, checkpoint_path, representations_path).
    """
    ensure_dirs()
    output_dir = output_dir or os.path.join(CHECKPOINT_DIR, "b2_adversarial")
    os.makedirs(output_dir, exist_ok=True)
    device = device or get_device()

    model = QwenAdversarialModel(
        model_name=model_name,
        num_task_labels=num_labels,
        num_bias_labels=num_bias_labels,
    )
    if use_lora:
        apply_lora_to_adversarial_backbone(model, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        print(
            f"B2 LoRA: r={lora_r} alpha={lora_alpha} (base backbone frozen; adapters + heads train) | "
            f"bias_head=MLP | bias_loss_balance={balance_bias_loss}"
        )
    else:
        print(f"B2 full backbone + heads (AdamW) | bias_head=MLP | bias_loss_balance={balance_bias_loss}")
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    pin_memory = device.startswith("cuda") if isinstance(device, str) else False
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
        pin_memory=pin_memory,
    )

    model.train()
    n_batches = max(len(train_loader), 1)
    for epoch in range(epochs):
        total_loss = 0.0
        sum_task = 0.0
        sum_bias = 0.0
        sum_bias_term = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
                "bias_labels": torch.tensor(batch["sensitive_attribute"], dtype=torch.long, device=device),
            }
            out = model(**inputs)
            L_task = out["loss_task"]
            L_bias = out["loss_bias"]
            b_term = bias_loss_term(
                L_task, L_bias, lambda_bias, balance_magnitudes=balance_bias_loss
            )
            loss = L_task + b_term
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            sum_task += float(L_task.detach().item())
            sum_bias += float(L_bias.detach().item())
            sum_bias_term += float(b_term.detach().item())
        mt = sum_task / n_batches
        mb = sum_bias / n_batches
        mterm = sum_bias_term / n_batches
        ratio = mt / (mb + 1e-9)
        print(
            f"B2 Epoch {epoch + 1}/{epochs}  loss_avg={total_loss / n_batches:.4f}  "
            f"L_task_avg={mt:.4f}  L_bias_avg={mb:.4f}  (λ1·L_bias)_term_avg={mterm:.4f}  "
            f"ratio_L_task/L_bias={ratio:.2f}"
        )

    ckpt_path = os.path.join(output_dir, "pytorch_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "model_name": model_name,
            "num_labels": num_labels,
            "num_bias_labels": num_bias_labels,
            "lambda_bias": lambda_bias,
            "use_lora": use_lora,
            "lora_r": lora_r if use_lora else None,
            "lora_alpha": lora_alpha if use_lora else None,
            "lora_dropout": lora_dropout if use_lora else None,
            "balance_bias_loss": balance_bias_loss,
        },
    }, ckpt_path)
    print(f"B2 checkpoint saved to {ckpt_path}")

    reps_path = None
    if save_representations:
        reps_dir = os.path.join(REPRESENTATIONS_DIR, "b2_adversarial")
        os.makedirs(reps_dir, exist_ok=True)
        model.eval()
        all_hidden, all_sensitive = [], []
        with torch.no_grad():
            for batch in train_loader:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                all_hidden.append(out["hidden_states"].cpu())
                all_sensitive.extend(batch["sensitive_attribute"])
        torch.save({
            "hidden_states": torch.cat(all_hidden, dim=0),
            "sensitive_attributes": all_sensitive,
        }, os.path.join(reps_dir, "hidden_and_metadata.pt"))
        reps_path = os.path.join(reps_dir, "hidden_and_metadata.pt")
    return model, ckpt_path, reps_path
