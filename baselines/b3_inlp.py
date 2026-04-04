"""
Baseline B3: INLP (Iterative Nullspace Projection).
1. Extract h from pretrained Qwen.
2. Train linear classifier to predict s, get W.
3. Compute projection P that removes W.
4. Apply P to representations.
5. Repeat k times.
6. Train task head on projected representations.
No adversarial training, no stability objective. Returns projected model.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List
from datasets import Dataset
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CHECKPOINT_DIR, REPRESENTATIONS_DIR, DEFAULT_INLP_ITERATIONS, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR, PROBE_MAX_ITER, ensure_dirs, get_device
from data.loaders import get_qwen_tokenizer
from baselines.b1_standard import _collate_batch
from models.qwen_task import QwenTaskModel


def _get_weight_vector(clf: LogisticRegression, scaler: Optional[StandardScaler] = None) -> np.ndarray:
    """Return weight vector w (feature_dim,) in original feature space. If scaler given, unscale coef."""
    coef = clf.coef_
    if coef.shape[0] == 1:
        w = coef[0].copy()
    else:
        w = coef[0].copy()
    if scaler is not None and hasattr(scaler, "scale_"):
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        scale[scale < 1e-12] = 1.0
        w = w / scale
    return w


def _nullspace_projection(w: np.ndarray) -> np.ndarray:
    """P such that P @ h removes component along w. P = I - (w'w)^{-1} w w'."""
    w = np.asarray(w, dtype=np.float64).ravel()
    w = w / (np.linalg.norm(w) + 1e-12)
    P = np.eye(len(w)) - np.outer(w, w)
    return P.astype(np.float32)


def _iterate_inlp(
    H: np.ndarray,
    s: np.ndarray,
    k: int = 10,
    random_state: int = 42,
    max_iter: int = None,
) -> tuple:
    """Run k INLP iterations. H (n, d), s (n,). Return (projected H, P_total). Scale X for stable LR fit."""
    if max_iter is None:
        max_iter = PROBE_MAX_ITER
    X = np.asarray(H, dtype=np.float64)
    y = np.asarray(s).ravel()
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    P_total = np.eye(X.shape[1], dtype=np.float64)
    for _ in range(k):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=max_iter, random_state=random_state, solver="lbfgs")
        clf.fit(X_scaled, y)
        w = _get_weight_vector(clf, scaler)
        if w.shape[0] != X.shape[1]:
            break
        P_i = _nullspace_projection(w)
        P_total = P_i @ P_total
        X = (P_total @ X.T).T
    return X.astype(np.float32), P_total.astype(np.float32)


def run_b3(
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    model_name: str = "Qwen/Qwen2.5-0.5B",
    num_labels: int = 2,
    num_bias_labels: int = 2,
    k_iterations: int = DEFAULT_INLP_ITERATIONS,
    output_dir: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    device: Optional[str] = None,
    save_representations: bool = True,
) -> tuple:
    """
    Run INLP: extract h, iterate nullspace projection k times, train task head on projected h.
    Returns (model with P baked in, checkpoint_path, representations_path).
    """
    ensure_dirs()
    output_dir = output_dir or os.path.join(CHECKPOINT_DIR, "b3_inlp")
    os.makedirs(output_dir, exist_ok=True)
    device = device or get_device()

    # 1. Extract hidden representations from pretrained Qwen (no task head yet)
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(model_name)
    backbone = AutoModelForCausalLM.from_pretrained(model_name, config=config).to(device)
    backbone.eval()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )
    all_h = []
    all_s = []
    with torch.no_grad():
        for batch in train_loader:
            out = backbone(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                output_hidden_states=True,
            )
            hidden = out.hidden_states[-1]
            mask = batch["attention_mask"].to(device).unsqueeze(-1).float()
            h = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_h.append(h.cpu().numpy())
            all_s.extend(batch["sensitive_attribute"])
    H = np.concatenate(all_h, axis=0)
    s = np.array(all_s)

    # 2–5. INLP iterations
    H_proj, P = _iterate_inlp(H, s, k=k_iterations)
    P_tensor = torch.from_numpy(P)

    # 6. Train task head on projected representations
    model = QwenTaskModel(
        model_name=model_name,
        num_labels=num_labels,
        projection_matrix=P_tensor,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Freeze backbone, train only task head
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out["loss"].backward()
            optimizer.step()
            total_loss += out["loss"].item()
        print(f"B3 Epoch {epoch + 1}/{epochs} loss: {total_loss / len(train_loader):.4f}")

    ckpt_path = os.path.join(output_dir, "pytorch_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "projection_matrix": P_tensor,
        "config": {"model_name": model_name, "num_labels": num_labels, "k_iterations": k_iterations},
    }, ckpt_path)

    if save_representations:
        reps_dir = os.path.join(REPRESENTATIONS_DIR, "b3_inlp")
        os.makedirs(reps_dir, exist_ok=True)
        torch.save({
            "hidden_states": torch.from_numpy(H_proj),
            "sensitive_attributes": all_s,
        }, os.path.join(reps_dir, "hidden_and_metadata.pt"))
    return model, ckpt_path, os.path.join(REPRESENTATIONS_DIR, "b3_inlp", "hidden_and_metadata.pt") if save_representations else None
