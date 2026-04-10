"""
Main model: stability-regularized bias suppression (extends B2 adversarial + LoRA).

L = L_task + λ1 * L_bias + λ2 * L_stab

L_stab penalizes drift in bias-head logits after a *simulated* task-only adaptation step on a
deep-copied model (LoRA + task head updated; bias head frozen). Gradients flow through the main
model via s_pred; s_pred_tilde is detached (no backprop through the copy).

Model definition: ``QwenAdversarialModel`` in ``models/adversarial.py`` (shared Qwen backbone +
task head + GRL bias head).
Forward contract: ``out = model(...)`` returns a dict with
``logits``, ``bias_logits``, optional ``loss_task``/``loss_bias``, and pooled ``hidden_states``.
"""
from __future__ import annotations

import copy
import gc
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
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
    DEFAULT_LAMBDA_STAB,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    ensure_dirs,
    get_device,
)
from baselines.b1_standard import _collate_batch
from baselines.b2_adversarial import apply_lora_to_adversarial_backbone, load_b2_from_checkpoint
from models.adversarial import QwenAdversarialModel, bias_loss_term

# Stabilize inner adaptation + KL stab loss (NaNs otherwise common with multi-step inner + high λ1)
MAIN_LOGIT_CLAMP = 35.0
MAIN_GRAD_CLIP_NORM = 1.0


def _inner_sgd_params(model_inner: QwenAdversarialModel, model: QwenAdversarialModel) -> List[torch.nn.Parameter]:
    """
    Freeze all of ``model_inner`` (no .grad buffers on frozen weights), then enable grad only on
    parameters that are trainable on ``model`` except ``bias_head`` (inner loop is task-only).
    """
    for p in model_inner.parameters():
        p.requires_grad = False
    for (nm, pm), (ni, pi) in zip(model.named_parameters(), model_inner.named_parameters()):
        if nm != ni:
            raise RuntimeError(f"model / model_inner param name mismatch: {nm!r} vs {ni!r}")
        if "bias_head" in nm:
            continue
        if pm.requires_grad:
            pi.requires_grad = True
    inner = [p for p in model_inner.parameters() if p.requires_grad]
    if not inner:
        raise RuntimeError("Main inner loop has no trainable parameters (check LoRA / task head).")
    return inner


def stability_loss(
    bias_logits_before: torch.Tensor,
    bias_logits_after: torch.Tensor,
    mode: str = "kl",
) -> torch.Tensor:
    """
    Drift between bias-head logits before and after simulated adaptation.
    Gradients should flow through `bias_logits_before`; `bias_logits_after` is detached upstream.
    """
    lo, hi = -MAIN_LOGIT_CLAMP, MAIN_LOGIT_CLAMP
    if mode == "mse":
        return F.mse_loss(
            bias_logits_before.clamp(lo, hi),
            bias_logits_after.clamp(lo, hi),
        )
    if mode == "kl":
        # KL(P || Q) with P from before (grad), Q from after (detached). Clamp logits for finite softmax / KL.
        b1 = bias_logits_before.clamp(lo, hi)
        b2 = bias_logits_after.detach().clamp(lo, hi)
        log_p = F.log_softmax(b1, dim=-1)
        log_q = F.log_softmax(b2, dim=-1)
        p = log_p.exp().clamp(min=1e-8)
        return (p * (log_p - log_q)).sum(dim=-1).mean()
    raise ValueError(f"Unknown stability mode: {mode}")


def run_main(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,  # reserved / unused (API parity with other baselines)
    model_name: str = "Qwen/Qwen2.5-0.5B",
    num_labels: int = 28,
    num_bias_labels: int = 2,
    lambda_bias: float = DEFAULT_LAMBDA_BIAS,
    lambda_stab: float = DEFAULT_LAMBDA_STAB,
    inner_lr: Optional[float] = None,
    inner_steps: int = 1,
    stab_loss_mode: str = "kl",
    output_dir: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    device: Optional[str] = None,
    save_representations: bool = False,
    use_gradient_checkpointing: bool = True,
    use_lora: bool = True,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    grad_clip_norm: float = MAIN_GRAD_CLIP_NORM,
    balance_bias_loss: bool = DEFAULT_ADV_BIAS_LOSS_BALANCE,
) -> tuple:
    """
    Train the Main model: adversarial debiasing + stability regularizer.

    Per batch:
      1) Forward main model → L_task, L_bias, bias logits (before).
      2) Sync a **GPU** twin from current weights when training on CUDA. All twin weights are frozen
         for autograd except the same subset the outer model trains excluding ``bias_head``, so the twin
         holds **weights only** on frozen modules (no grad tensors there). On CPU-only runs, twin stays on CPU.
      3) Forward copy → bias logits (after); L_stab = MSE or KL between before/after.
      4) L = L_task + λ1 L_bias + λ2 L_stab; backward on main model only.

    ``save_representations``: if False (default), skips saving train-set hidden states (saves RAM).

    ``use_gradient_checkpointing``: if True (default), enables HF gradient checkpointing on the backbone
    (lower activation VRAM, somewhat slower steps).

    Returns (model, checkpoint_path, representations_path or None).
    """
    if inner_steps < 1:
        raise ValueError("inner_steps must be >= 1")
    if stab_loss_mode not in ("kl", "mse"):
        raise ValueError("stab_loss_mode must be 'kl' or 'mse'")

    ensure_dirs()
    output_dir = output_dir or os.path.join(CHECKPOINT_DIR, "main")
    os.makedirs(output_dir, exist_ok=True)
    device = device or get_device()
    inner_lr = inner_lr if inner_lr is not None else lr

    model = QwenAdversarialModel(
        model_name=model_name,
        num_task_labels=num_labels,
        num_bias_labels=num_bias_labels,
    )
    if use_lora:
        apply_lora_to_adversarial_backbone(
            model, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        print(
            f"Main (LoRA): r={lora_r} alpha={lora_alpha} | "
            f"λ1={lambda_bias} λ2={lambda_stab} inner_lr={inner_lr} steps={inner_steps} stab={stab_loss_mode} | "
            f"bias_loss_balance={balance_bias_loss}"
        )
    else:
        print(
            f"Main (full FT, no LoRA): λ1={lambda_bias} λ2={lambda_stab} "
            f"inner_lr={inner_lr} steps={inner_steps} stab={stab_loss_mode} | "
            f"bias_loss_balance={balance_bias_loss}"
        )

    model = model.to(device)
    dev = torch.device(device)
    if use_gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()
        print("Main: gradient checkpointing enabled on backbone (lower activation memory).")

    trainable_main = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_main, lr=lr)

    if dev.type == "cuda":
        model_inner = copy.deepcopy(model).to(dev)
        torch.cuda.empty_cache()
        print("Main: inner (simulated) adaptation runs on GPU (twin model in VRAM).")
    else:
        model_inner = copy.deepcopy(model)

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
        sum_bterm = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
                "bias_labels": torch.tensor(batch["sensitive_attribute"], dtype=torch.long, device=device),
            }
            out = model(**inputs)
            l_task = out["loss_task"]
            l_bias = out["loss_bias"]
            s_pred = out["bias_logits"]
            del out

            # load_state_dict copies into model_inner's parameters — no extra full state_dict clone dict
            model_inner.load_state_dict(model.state_dict(), strict=True)
            model_inner.train()
            inner_params = _inner_sgd_params(model_inner, model)
            inner_opt = torch.optim.SGD(inner_params, lr=inner_lr, momentum=0.0)

            if dev.type == "cuda":
                ids_c = inputs["input_ids"]
                mask_c = inputs["attention_mask"]
                labels_c = inputs["labels"]
            else:
                ids_c = inputs["input_ids"].cpu()
                mask_c = inputs["attention_mask"].cpu()
                labels_c = inputs["labels"].cpu()

            with torch.enable_grad():
                for _ in range(inner_steps):
                    inner_opt.zero_grad()
                    inner_out = model_inner(
                        input_ids=ids_c,
                        attention_mask=mask_c,
                        labels=labels_c,
                    )
                    inner_lt = inner_out["loss_task"]
                    if not torch.isfinite(inner_lt):
                        break
                    inner_lt.backward()
                    torch.nn.utils.clip_grad_norm_(inner_params, max_norm=grad_clip_norm)
                    inner_opt.step()

            model_inner.eval()
            with torch.no_grad():
                after_out = model_inner(
                    input_ids=ids_c,
                    attention_mask=mask_c,
                )
            s_pred_tilde = after_out["bias_logits"].to(dev).detach()
            if not torch.isfinite(s_pred_tilde).all():
                l_stab = torch.zeros((), device=device, dtype=l_task.dtype)
            else:
                l_stab = stability_loss(s_pred, s_pred_tilde, mode=stab_loss_mode)

            b_term = bias_loss_term(
                l_task, l_bias, lambda_bias, balance_magnitudes=balance_bias_loss
            )
            loss = l_task + b_term + lambda_stab * l_stab
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_main, max_norm=grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.detach().item())
            sum_task += float(l_task.detach().item())
            sum_bias += float(l_bias.detach().item())
            sum_bterm += float(b_term.detach().item())
            gc.collect()

        mt = sum_task / n_batches
        mb = sum_bias / n_batches
        mbterm = sum_bterm / n_batches
        ratio = mt / (mb + 1e-9)
        print(
            f"Main Epoch {epoch + 1}/{epochs}  loss_avg={total_loss / n_batches:.4f}  "
            f"L_task_avg={mt:.4f}  L_bias_avg={mb:.4f}  (λ1·L_bias)_term_avg={mbterm:.4f}  "
            f"ratio_L_task/L_bias={ratio:.2f}"
        )

    del model_inner
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    ckpt_path = os.path.join(output_dir, "pytorch_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_name": model_name,
                "num_labels": num_labels,
                "num_bias_labels": num_bias_labels,
                "lambda_bias": lambda_bias,
                "lambda_stab": lambda_stab,
                "inner_lr": inner_lr,
                "inner_steps": inner_steps,
                "stab_loss_mode": stab_loss_mode,
                "use_lora": use_lora,
                "lora_r": lora_r if use_lora else None,
                "lora_alpha": lora_alpha if use_lora else None,
                "lora_dropout": lora_dropout if use_lora else None,
                "baseline": "main_stability",
                "balance_bias_loss": balance_bias_loss,
                "inner_adapt_on_gpu": dev.type == "cuda",
                "gradient_checkpointing": bool(use_gradient_checkpointing),
            },
        },
        ckpt_path,
    )
    print(f"Main checkpoint saved to {ckpt_path}")

    reps_path = None
    if save_representations:
        reps_dir = os.path.join(REPRESENTATIONS_DIR, "main")
        os.makedirs(reps_dir, exist_ok=True)
        model.eval()
        all_hidden, all_sensitive = [], []
        with torch.no_grad():
            for b in train_loader:
                o = model(
                    input_ids=b["input_ids"].to(device),
                    attention_mask=b["attention_mask"].to(device),
                )
                all_hidden.append(o["hidden_states"].cpu())
                all_sensitive.extend(b["sensitive_attribute"])
        torch.save(
            {
                "hidden_states": torch.cat(all_hidden, dim=0),
                "sensitive_attributes": all_sensitive,
            },
            os.path.join(reps_dir, "hidden_and_metadata.pt"),
        )
        reps_path = os.path.join(reps_dir, "hidden_and_metadata.pt")

    return model, ckpt_path, reps_path


def load_main_from_checkpoint(
    ckpt_path: str,
    model_name: str,
    device: str,
    num_task_labels: int,
    num_bias_labels: int = 2,
) -> QwenAdversarialModel:
    """Load Main model checkpoint (same architecture / LoRA pattern as B2)."""
    return load_b2_from_checkpoint(ckpt_path, model_name, device, num_task_labels, num_bias_labels)
