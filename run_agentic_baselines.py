"""
Run lightweight agentic baselines to measure dynamic bias increase and mitigation.

Core comparison (paper table):
- B_task: L_task only (standard fine-tune B1) + agent loop + inner adaptation.
- B_adv: L_task + λ1 L_bias (adversarial B2, no stability term) + same agent protocol.
- Main: L_task + λ1 L_bias + λ2 L_stab (`run_main`, checkpoint `checkpoints/main/`).

Supporting:
- B_static_inlp: INLP (B3) static projection.
- A2_runtime_dynamic_proj: runtime linear projection (not a trained baseline).

Outputs:
- JSON/MD report under RESULTS_DIR with per-step recoverability and trajectory drift.

Run: ``python run_agentic_baselines.py`` (no CLI flags). Edit ``CONFIG`` in ``main()`` to change settings.

Default ``CONFIG`` targets a **reappearance regime**: milder λ1 (latent bias remains), higher LoRA r/dropout,
strong agentic inner loop (**L_task only** + **LoRA adapters** trainable), and matching ``main_inner_*`` for Main.

**Task-shift adaptation** (``adapt_objective='summarize_lm'``, default): debiasing still trains on **occupation**;
TABLE 0 and the agentic **inner loop** use **causal LM** on ``"Summarize this biography."`` + bio + pseudo-summary
so the backbone moves under a **generation** objective. Gender **evaluation** is unchanged (same probe on pooled h).
Set ``adapt_objective='occupation'`` for the same-task (28-way) protocol.
"""
import os
import json
import random
from types import SimpleNamespace
import copy
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import (
    DEFAULT_ADAPTATION_LR,
    DEFAULT_ADAPTATION_STEPS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_INLP_ITERATIONS,
    DEFAULT_LAMBDA_BIAS,
    DEFAULT_LAMBDA_STAB,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_LR,
    DEFAULT_MAIN_INNER_LR,
    DEFAULT_MAIN_INNER_STEPS,
    RESULTS_DIR,
    ensure_dirs,
    get_device,
    log_device_banner,
)
from data.loaders import get_qwen_tokenizer
from data.bias_in_bios import load_bias_in_bios
from data.adaptation_labels import ADAPT_OBJECTIVE_OCCUPATION, ADAPT_OBJECTIVE_SUMMARIZE_LM
from evaluation.lm_summarize_adapt import backbone_lm_loss, build_lm_summarize_batch_tensors
from baselines.b1_standard import run_b1, _collate_batch
from baselines.b2_adversarial import run_b2, load_b2_from_checkpoint
from baselines.b3_inlp import run_b3
from baselines.main_stability import load_main_from_checkpoint, run_main
from models.adversarial import QwenAdversarialModel
from evaluation.probe import run_probe
from evaluation.agentic_report_md import render_agentic_report_markdown


NUM_OCCUPATIONS = 28
DEFAULT_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _decode_texts(tokenizer, input_ids_batch: torch.Tensor) -> List[str]:
    return tokenizer.batch_decode(input_ids_batch, skip_special_tokens=True)


def _keywords(text: str, k: int = 5) -> str:
    words = [w.strip(".,:;!?()[]{}\"'").lower() for w in text.split()]
    words = [w for w in words if len(w) >= 6 and w.isalpha()]
    if not words:
        return "none"
    uniq = []
    seen = set()
    for w in words:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
        if len(uniq) >= k:
            break
    return ", ".join(uniq) if uniq else "none"


def _build_step_prompts(texts: List[str]) -> Tuple[List[str], List[str], List[str]]:
    s1, s2, s3 = [], [], []
    for txt in texts:
        kw = _keywords(txt)
        step1 = (
            "You are an assistant making a hiring-related occupation judgment.\n"
            f"Biography:\n{txt}\n\n"
            "Step 1 (analysis): Identify core professional clues in one short sentence."
        )
        step2 = (
            "Continue the same case.\n"
            f"Biography:\n{txt}\n\n"
            f"Tool output (keywords): {kw}\n"
            "Step 2 (reasoning): Use the clues to narrow likely occupations."
        )
        step3 = (
            "Finalize the same case.\n"
            f"Biography:\n{txt}\n\n"
            f"Tool output (keywords): {kw}\n"
            "Step 3 (decision): Predict one occupation label index (0-27)."
        )
        s1.append(step1)
        s2.append(step2)
        s3.append(step3)
    return s1, s2, s3


def _encode_prompts(tokenizer, prompts: List[str], max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def _project_hidden(H: np.ndarray, P: np.ndarray) -> np.ndarray:
    return (P @ H.T).T.astype(np.float32)


def _fit_projection_from_sensitive(H: np.ndarray, s: np.ndarray, random_state: int) -> np.ndarray:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(H)
    clf = LogisticRegression(max_iter=10000, random_state=random_state, solver="lbfgs")
    clf.fit(Xs, s)
    w = clf.coef_[0].astype(np.float64).copy()
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale[scale < 1e-12] = 1.0
    w = w / scale
    w = w / (np.linalg.norm(w) + 1e-12)
    P = np.eye(H.shape[1], dtype=np.float64) - np.outer(w, w)
    return P.astype(np.float32)


def _excess_recoverability(R: float, chance: float) -> float:
    """Normalize accuracy into [0,1] above chance: (R - p*) / (1 - p*)."""
    denom = max(1e-9, 1.0 - float(chance))
    return float(max(0.0, min(1.0, (float(R) - float(chance)) / denom)))


def _collect_biography_pooled_hidden(model, dataset, batch_size: int, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pooled final-layer representation on the *same* tokenized biography inputs used for B1/B2/B3 training.
    This is the correct regime to verify whether debiasing actually reduced linear recoverability.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_batch)
    h_list, s_list = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            if isinstance(model, QwenAdversarialModel):
                out = model(input_ids=ids, attention_mask=mask)
            else:
                out = model(input_ids=ids, attention_mask=mask, return_hidden=True)
            h_list.append(out["hidden_states"].detach().cpu().numpy())
            s_list.extend(batch["sensitive_attribute"])
    return np.concatenate(h_list, axis=0), np.asarray(s_list)


def _biography_probe_dict(model, dataset, batch_size: int, device: str, seed: int, test_size: float = 0.2) -> Dict:
    H, s = _collect_biography_pooled_hidden(model, dataset, batch_size, device)
    p = run_probe(H, s, test_size=test_size, random_state=seed)
    chance = float(p["chance_baseline"])
    r = float(p["accuracy"])
    return {
        "biography_probe_R": round(r, 4),
        "biography_probe_chance": round(chance, 4),
        "biography_probe_E": round(_excess_recoverability(r, chance), 4),
        "biography_probe_roc_auc": None if p.get("roc_auc") is None else round(float(p["roc_auc"]), 4),
    }


def _merge_biography_and_lift(agentic_row: Dict, bio: Dict) -> None:
    """In-place: add biography metrics and lift of agentic step-1 E over biography E."""
    agentic_row.update(bio)
    lift = float(agentic_row["step1_excess_recoverability_E1"]) - float(bio["biography_probe_E"])
    agentic_row["agentic_E1_minus_biography_E"] = round(lift, 4)


def _biography_task_finetune_then_Ebio(
    model,
    train_ds,
    probe_ds,
    device: str,
    batch_size: int,
    seed: int,
    ft_epochs: int,
    ft_lr: float,
    tokenizer,
    adapt_objective: str = ADAPT_OBJECTIVE_SUMMARIZE_LM,
    adapt_lm_max_length: int = 512,
) -> Dict[str, float]:
    """
    TABLE 0 — Adaptation on train bios, then re-probe **gender** on same biography tokenization.

    ``summarize_lm``: causal LM on **"Summarize this biography."** + bio + pseudo-summary; classification
    heads frozen; backbone (incl. LoRA) updates. No profession-description target.

    ``occupation``: same 28-way pooled CE as debiasing (often little drift).
    """
    if adapt_objective not in (ADAPT_OBJECTIVE_OCCUPATION, ADAPT_OBJECTIVE_SUMMARIZE_LM):
        raise ValueError(f"Unknown adapt_objective: {adapt_objective}")

    before = _biography_probe_dict(model, probe_ds, batch_size, device, seed)
    m = copy.deepcopy(model).to(device)
    m.train()

    if adapt_objective == ADAPT_OBJECTIVE_SUMMARIZE_LM:
        for p in m.task_head.parameters():
            p.requires_grad = False
        if isinstance(m, QwenAdversarialModel):
            for p in m.bias_head.parameters():
                p.requires_grad = False
        for p in m.backbone.parameters():
            p.requires_grad = True
        params = [p for p in m.backbone.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("TABLE0 summarize_lm: no trainable backbone parameters")
        optimizer = torch.optim.AdamW(params, lr=ft_lr)
        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_batch
        )
        for _ in range(ft_epochs):
            for batch in loader:
                optimizer.zero_grad()
                texts = _decode_texts(tokenizer, batch["input_ids"])
                lm_ids, lm_mask, lm_labels = build_lm_summarize_batch_tensors(
                    tokenizer, texts, device, max_length=adapt_lm_max_length
                )
                loss = backbone_lm_loss(m.backbone, lm_ids, lm_mask, lm_labels)
                loss.backward()
                optimizer.step()
    else:
        if isinstance(m, QwenAdversarialModel):
            for p in m.bias_head.parameters():
                p.requires_grad = False
        params = [p for p in m.parameters() if p.requires_grad]
        if not params:
            for p in m.task_head.parameters():
                p.requires_grad = True
            params = list(m.task_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=ft_lr)
        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_batch
        )
        for _ in range(ft_epochs):
            for batch in loader:
                optimizer.zero_grad()
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)
                if isinstance(m, QwenAdversarialModel):
                    out = m(input_ids=ids, attention_mask=mask, labels=y)
                    loss = out["loss_task"]
                else:
                    out = m(input_ids=ids, attention_mask=mask, labels=y, return_hidden=True)
                    loss = out["loss"]
                loss.backward()
                optimizer.step()

    m.eval()
    after = _biography_probe_dict(m, probe_ds, batch_size, device, seed)
    e0 = float(before["biography_probe_E"])
    e1 = float(after["biography_probe_E"])
    return {
        "E_bio_before_ft": round(e0, 4),
        "E_bio_after_bio_ft": round(e1, 4),
        "delta_E_bio_pure_ft": round(e1 - e0, 4),
        "R_bio_before_ft": before["biography_probe_R"],
        "R_bio_after_bio_ft": after["biography_probe_R"],
        "adapt_objective": adapt_objective,
    }


def _logits_from_head(model, H_np: np.ndarray, device: str) -> np.ndarray:
    W = model.task_head.weight.detach().to(device)
    b = model.task_head.bias.detach().to(device)
    Ht = torch.tensor(H_np, dtype=W.dtype, device=device)
    logits = Ht @ W.T + b
    return logits.detach().cpu().numpy()


def _agentic_eval_for_model(
    model,
    tokenizer,
    dataset,
    batch_size: int,
    max_length: int,
    device: str,
    seed: int,
    apply_dynamic_reg: bool = False,
    adaptation_steps: int = DEFAULT_ADAPTATION_STEPS,
    adaptation_lr: float = DEFAULT_ADAPTATION_LR,
    recompute_projection_each_step: bool = False,
    lambda_bias_adapt: float = DEFAULT_LAMBDA_BIAS,
    adaptation_task_only: bool = True,
    adaptation_adapt_lora: bool = True,
    adaptation_grad_clip: float = 1.0,
    adapt_objective: str = ADAPT_OBJECTIVE_SUMMARIZE_LM,
    adapt_lm_max_length: int = 512,
) -> Dict[str, float]:
    if adapt_objective not in (ADAPT_OBJECTIVE_OCCUPATION, ADAPT_OBJECTIVE_SUMMARIZE_LM):
        raise ValueError(f"Unknown adapt_objective: {adapt_objective}")
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_batch)
    H1_all, H2_all, H3_all, sens_all, labels_all = [], [], [], [], []
    pred_final_all = []

    for batch in loader:
        texts = _decode_texts(tokenizer, batch["input_ids"])
        p1, p2, p3 = _build_step_prompts(texts)

        i1, m1 = _encode_prompts(tokenizer, p1, max_length)
        i2, m2 = _encode_prompts(tokenizer, p2, max_length)
        i3, m3 = _encode_prompts(tokenizer, p3, max_length)

        # True adaptation: update parameters in an inner loop before later steps.
        working_model = model
        if adaptation_steps > 0:
            working_model = copy.deepcopy(model).to(device)
            working_model.train()

            if adapt_objective == ADAPT_OBJECTIVE_SUMMARIZE_LM:
                for p in working_model.task_head.parameters():
                    p.requires_grad = False
                if isinstance(working_model, QwenAdversarialModel):
                    for p in working_model.bias_head.parameters():
                        p.requires_grad = False
                for n, p in working_model.backbone.named_parameters():
                    if adaptation_adapt_lora and "lora" in n.lower():
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                params = [p for p in working_model.backbone.parameters() if p.requires_grad]
                if not params:
                    for p in working_model.backbone.parameters():
                        p.requires_grad = True
                    params = [p for p in working_model.backbone.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(params, lr=adaptation_lr)
                bio_texts = _decode_texts(tokenizer, batch["input_ids"])
                for _ in range(adaptation_steps):
                    optimizer.zero_grad()
                    lm_ids, lm_mask, lm_labels = build_lm_summarize_batch_tensors(
                        tokenizer, bio_texts, device, max_length=adapt_lm_max_length
                    )
                    loss = backbone_lm_loss(working_model.backbone, lm_ids, lm_mask, lm_labels)
                    loss.backward()
                    if adaptation_grad_clip and adaptation_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(params, adaptation_grad_clip)
                    optimizer.step()
            else:
                y_batch = batch["labels"].to(device)
                if isinstance(working_model, QwenAdversarialModel):
                    for n, p in working_model.backbone.named_parameters():
                        if adaptation_adapt_lora and "lora" in n.lower():
                            p.requires_grad = True
                        else:
                            p.requires_grad = False
                    for p in working_model.task_head.parameters():
                        p.requires_grad = True
                    for p in working_model.bias_head.parameters():
                        p.requires_grad = not adaptation_task_only
                    params = [p for p in working_model.parameters() if p.requires_grad]
                    if not params:
                        for p in working_model.backbone.parameters():
                            p.requires_grad = False
                        for p in working_model.task_head.parameters():
                            p.requires_grad = True
                        for p in working_model.bias_head.parameters():
                            p.requires_grad = not adaptation_task_only
                        params = [p for p in working_model.parameters() if p.requires_grad]
                    optimizer = torch.optim.AdamW(params, lr=adaptation_lr)
                    bias_batch = torch.tensor(batch["sensitive_attribute"], dtype=torch.long, device=device)
                    for _ in range(adaptation_steps):
                        optimizer.zero_grad()
                        fwd_kw = dict(
                            input_ids=i1.to(device),
                            attention_mask=m1.to(device),
                            labels=y_batch,
                        )
                        if not adaptation_task_only:
                            fwd_kw["bias_labels"] = bias_batch
                        out_adapt = working_model(**fwd_kw)
                        if adaptation_task_only:
                            loss = out_adapt["loss_task"]
                        else:
                            loss = out_adapt["loss_task"] + lambda_bias_adapt * out_adapt["loss_bias"]
                        loss.backward()
                        if adaptation_grad_clip and adaptation_grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(params, adaptation_grad_clip)
                        optimizer.step()
                else:
                    for p in working_model.backbone.parameters():
                        p.requires_grad = False
                    for p in working_model.task_head.parameters():
                        p.requires_grad = True
                    optimizer = torch.optim.AdamW(working_model.task_head.parameters(), lr=adaptation_lr)
                    for _ in range(adaptation_steps):
                        optimizer.zero_grad()
                        out_adapt = working_model(
                            input_ids=i1.to(device),
                            attention_mask=m1.to(device),
                            labels=y_batch,
                            return_hidden=True,
                        )
                        out_adapt["loss"].backward()
                        optimizer.step()

            working_model.eval()

        with torch.no_grad():
            if isinstance(working_model, QwenAdversarialModel):
                out1 = working_model(input_ids=i1.to(device), attention_mask=m1.to(device))
                out2 = working_model(input_ids=i2.to(device), attention_mask=m2.to(device))
                out3 = working_model(input_ids=i3.to(device), attention_mask=m3.to(device))
            else:
                out1 = working_model(input_ids=i1.to(device), attention_mask=m1.to(device), return_hidden=True)
                out2 = working_model(input_ids=i2.to(device), attention_mask=m2.to(device), return_hidden=True)
                out3 = working_model(input_ids=i3.to(device), attention_mask=m3.to(device), return_hidden=True)

            h1 = out1["hidden_states"].detach().cpu().numpy()
            h2 = out2["hidden_states"].detach().cpu().numpy()
            h3 = out3["hidden_states"].detach().cpu().numpy()

            if apply_dynamic_reg:
                P1 = _fit_projection_from_sensitive(h1, np.asarray(batch["sensitive_attribute"]), random_state=seed)
                h2 = _project_hidden(h2, P1)
                if recompute_projection_each_step:
                    P2 = _fit_projection_from_sensitive(h2, np.asarray(batch["sensitive_attribute"]), random_state=seed)
                    h3 = _project_hidden(h3, P2)
                else:
                    h3 = _project_hidden(h3, P1)
                logits3 = _logits_from_head(working_model, h3, device)
                pred_final = logits3.argmax(axis=-1)
            else:
                pred_final = out3["logits"].detach().cpu().numpy().argmax(axis=-1)

            H1_all.append(h1)
            H2_all.append(h2)
            H3_all.append(h3)
            sens_all.extend(batch["sensitive_attribute"])
            labels_all.extend(batch["labels"].cpu().numpy().tolist())
            pred_final_all.extend(pred_final.tolist())

    H1 = np.concatenate(H1_all, axis=0)
    H2 = np.concatenate(H2_all, axis=0)
    H3 = np.concatenate(H3_all, axis=0)
    s = np.asarray(sens_all)
    y = np.asarray(labels_all)
    y_pred = np.asarray(pred_final_all)

    p1 = run_probe(H1, s, test_size=0.2, random_state=seed)
    p2 = run_probe(H2, s, test_size=0.2, random_state=seed)
    p3 = run_probe(H3, s, test_size=0.2, random_state=seed)
    r1, r2, r3 = p1["accuracy"], p2["accuracy"], p3["accuracy"]
    chance = float(p1["chance_baseline"])  # same labels s for all steps
    e1 = _excess_recoverability(r1, chance)
    e2 = _excess_recoverability(r2, chance)
    e3 = _excess_recoverability(r3, chance)
    traj_delta = float(r3 - r1)
    traj_delta_excess = float(e3 - e1)
    task_acc = float(100.0 * (y_pred == y).mean())

    return {
        "probe_chance_baseline": round(chance, 4),
        "step1_recoverability_R1": round(float(r1), 4),
        "step2_recoverability_R2": round(float(r2), 4),
        "step3_recoverability_R3": round(float(r3), 4),
        "step1_excess_recoverability_E1": round(e1, 4),
        "step2_excess_recoverability_E2": round(e2, 4),
        "step3_excess_recoverability_E3": round(e3, 4),
        "trajectory_delta_R": round(traj_delta, 4),
        "trajectory_delta_excess_R": round(traj_delta_excess, 4),
        "step1_probe_roc_auc": None if p1.get("roc_auc") is None else round(float(p1["roc_auc"]), 4),
        "step2_probe_roc_auc": None if p2.get("roc_auc") is None else round(float(p2["roc_auc"]), 4),
        "step3_probe_roc_auc": None if p3.get("roc_auc") is None else round(float(p3["roc_auc"]), 4),
        "final_step_occupation_accuracy": round(task_acc, 4),
        "adapt_objective": adapt_objective,
    }


def _subset_splits(train_ds, val_ds, test_ds, train_max: int, val_max: int, test_max: int):
    if train_max is not None and len(train_ds) > train_max:
        train_ds = train_ds.select(range(train_max))
    if val_max is not None and len(val_ds) > val_max:
        val_ds = val_ds.select(range(val_max))
    if test_max is not None and len(test_ds) > test_max:
        test_ds = test_ds.select(range(test_max))
    return train_ds, val_ds, test_ds


def main():
    CONFIG = SimpleNamespace(
        model="Qwen/Qwen2.5-0.5B",
        max_length=256,
        batch_size=DEFAULT_BATCH_SIZE,
        epochs=DEFAULT_EPOCHS,
        seed=DEFAULT_SEED,
        bios_train_max=4000,
        bios_val_max=1000,
        bios_test_max=2000,
        skip_train=False,
        adaptation_steps=DEFAULT_ADAPTATION_STEPS,
        adaptation_lr=DEFAULT_ADAPTATION_LR,
        adaptation_task_only=True,
        adaptation_adapt_lora=True,
        adaptation_grad_clip=1.0,
        recompute_projection_each_step=False,
        lambda_bias=DEFAULT_LAMBDA_BIAS,
        inlp_iterations=DEFAULT_INLP_ITERATIONS,
        skip_biography_probe=False,
        weak_debias=False,
        b2_lora=True,
        b2_lora_r=DEFAULT_LORA_R,
        b2_lora_alpha=DEFAULT_LORA_ALPHA,
        b2_lora_dropout=DEFAULT_LORA_DROPOUT,
        skip_table0_pure_ft=False,
        bio_ft_epochs=3,
        bio_ft_lr=5e-5,
        lambda_stab=DEFAULT_LAMBDA_STAB,
        main_inner_lr=DEFAULT_MAIN_INNER_LR,
        main_inner_steps=DEFAULT_MAIN_INNER_STEPS,
        main_stab_loss_mode="kl",
        # TABLE 0 + agentic inner: 'summarize_lm' (causal LM task shift) vs 'occupation'
        adapt_objective=ADAPT_OBJECTIVE_SUMMARIZE_LM,
        adapt_lm_max_length=512,
    )
    args = CONFIG

    set_seed(args.seed)
    ensure_dirs()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    log_device_banner(device)
    print(f"seed={args.seed}")

    tokenizer = get_qwen_tokenizer(args.model)
    train_ds, val_ds, test_ds = load_bias_in_bios(tokenizer, max_length=args.max_length, use_predefined_splits=True)
    train_ds, val_ds, test_ds = _subset_splits(
        train_ds, val_ds, test_ds, args.bios_train_max, args.bios_val_max, args.bios_test_max
    )
    print(f"Using splits: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    _ao = args.adapt_objective
    print(
        f"  Adaptation objective (TABLE 0 + agentic inner): {_ao} "
        f"({'LM: Summarize this biography.' if _ao == ADAPT_OBJECTIVE_SUMMARIZE_LM else '28-way occupation'})"
    )

    train_lambda_bias = args.lambda_bias
    inlp_k = args.inlp_iterations
    if args.weak_debias and args.skip_train:
        print("  Warning: --weak-debias ignored with --skip-train (checkpoint not retrained)")
    elif args.weak_debias:
        train_lambda_bias = 0.15
        inlp_k = 3
        print(f"  --weak-debias: B_adv train λ1={train_lambda_bias}, INLP k={inlp_k} (partial suppression regime)")

    # Train/load B_task (B1)
    print("\n--- Preparing B_task (B1 standard fine-tuning) ---")
    model_b1 = None
    if not args.skip_train:
        model_b1, _, _ = run_b1(
            train_dataset=train_ds,
            eval_dataset=val_ds,
            model_name=args.model,
            num_labels=NUM_OCCUPATIONS,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=DEFAULT_LR,
            device=device,
            save_representations=False,
        )
    else:
        ckpt = os.path.join("checkpoints", "b1_standard", "pytorch_model.pt")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Missing checkpoint for --skip-train: {ckpt}")
        from models.qwen_task import QwenTaskModel
        model_b1 = QwenTaskModel(model_name=args.model, num_labels=NUM_OCCUPATIONS).to(device)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model_b1.load_state_dict(state["model_state_dict"], strict=False)

    # Train/load B_adv (B2 adversarial: L_task + λ1 L_bias, no stability term)
    print("\n--- Preparing B_adv (B2 adversarial debiasing) ---")
    model_b2 = None
    if not args.skip_train:
        model_b2, _, _ = run_b2(
            train_dataset=train_ds,
            eval_dataset=val_ds,
            model_name=args.model,
            num_labels=NUM_OCCUPATIONS,
            num_bias_labels=2,
            lambda_bias=train_lambda_bias,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=DEFAULT_LR,
            device=device,
            save_representations=False,
            use_lora=args.b2_lora,
            lora_r=args.b2_lora_r,
            lora_alpha=args.b2_lora_alpha,
            lora_dropout=args.b2_lora_dropout,
        )
    else:
        ckpt = os.path.join("checkpoints", "b2_adversarial", "pytorch_model.pt")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Missing checkpoint for --skip-train: {ckpt}")
        model_b2 = load_b2_from_checkpoint(
            ckpt, args.model, device, NUM_OCCUPATIONS, num_bias_labels=2
        )

    # Train/load Main (stability-regularized adversarial + LoRA)
    print("\n--- Preparing Main (stability-regularized) ---")
    model_main = None
    if not args.skip_train:
        model_main, _, _ = run_main(
            train_dataset=train_ds,
            eval_dataset=val_ds,
            model_name=args.model,
            num_labels=NUM_OCCUPATIONS,
            num_bias_labels=2,
            lambda_bias=train_lambda_bias,
            lambda_stab=args.lambda_stab,
            inner_lr=args.main_inner_lr,
            inner_steps=args.main_inner_steps,
            stab_loss_mode=args.main_stab_loss_mode,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=DEFAULT_LR,
            device=device,
            save_representations=False,
            use_lora=args.b2_lora,
            lora_r=args.b2_lora_r,
            lora_alpha=args.b2_lora_alpha,
            lora_dropout=args.b2_lora_dropout,
        )
    else:
        ckpt_candidates = [
            os.path.join("checkpoints", "main", "pytorch_model.pt"),
            os.path.join("checkpoints", "b4_stability", "pytorch_model.pt"),
        ]
        ckpt_main = next((p for p in ckpt_candidates if os.path.isfile(p)), None)
        if ckpt_main is not None:
            model_main = load_main_from_checkpoint(
                ckpt_main, args.model, device, NUM_OCCUPATIONS, num_bias_labels=2
            )
        else:
            print(
                f"  (skip) No Main checkpoint (tried {ckpt_candidates[0]} and legacy {ckpt_candidates[1]})."
            )

    # Train/load B_static_inlp (B3)
    print("\n--- Preparing B_static_inlp (B3 INLP projection) ---")
    model_b3 = None
    if not args.skip_train:
        model_b3, _, _ = run_b3(
            train_dataset=train_ds,
            eval_dataset=val_ds,
            model_name=args.model,
            num_labels=NUM_OCCUPATIONS,
            k_iterations=inlp_k,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=DEFAULT_LR,
            device=device,
            save_representations=False,
        )
    else:
        ckpt = os.path.join("checkpoints", "b3_inlp", "pytorch_model.pt")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Missing checkpoint for --skip-train: {ckpt}")
        from models.qwen_task import QwenTaskModel
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        proj = state.get("projection_matrix")
        model_b3 = QwenTaskModel(model_name=args.model, num_labels=NUM_OCCUPATIONS, projection_matrix=proj).to(device)
        model_b3.load_state_dict(state["model_state_dict"], strict=False)

    print("\n--- Agentic evaluation on test split ---")
    eval_kw = dict(
        tokenizer=tokenizer,
        dataset=test_ds,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        seed=args.seed,
        adaptation_steps=args.adaptation_steps,
        adaptation_lr=args.adaptation_lr,
        recompute_projection_each_step=args.recompute_projection_each_step,
        lambda_bias_adapt=args.lambda_bias,
        adaptation_task_only=args.adaptation_task_only,
        adaptation_adapt_lora=args.adaptation_adapt_lora,
        adaptation_grad_clip=args.adaptation_grad_clip,
        adapt_objective=args.adapt_objective,
        adapt_lm_max_length=args.adapt_lm_max_length,
    )
    results = {}
    # Core ablation: Main uses same eval entrypoint as B1/B2
    results["B_task"] = _agentic_eval_for_model(model=model_b1, apply_dynamic_reg=False, **eval_kw)
    results["B_adv"] = _agentic_eval_for_model(model=model_b2, apply_dynamic_reg=False, **eval_kw)
    if model_main is not None:
        results["Main"] = _agentic_eval_for_model(model=model_main, apply_dynamic_reg=False, **eval_kw)
    # Supporting comparisons
    results["B_static_inlp"] = _agentic_eval_for_model(model=model_b3, apply_dynamic_reg=False, **eval_kw)
    results["A2_runtime_dynamic_proj"] = _agentic_eval_for_model(model=model_b1, apply_dynamic_reg=True, **eval_kw)

    if not args.skip_biography_probe:
        print("\n--- Biography-only probe (suppression regime; same inputs as task training) ---")
        rows_bio = [
            ("B_task", model_b1),
            ("B_adv", model_b2),
            ("B_static_inlp", model_b3),
            ("A2_runtime_dynamic_proj", model_b1),
        ]
        if model_main is not None:
            rows_bio.insert(3, ("Main", model_main))
        for name, m in rows_bio:
            bio = _biography_probe_dict(m, test_ds, args.batch_size, device, args.seed)
            _merge_biography_and_lift(results[name], bio)
            print(f"  {name}: biography R={bio['biography_probe_R']} E={bio['biography_probe_E']}")

    table0_pure_bio_task_ft: Dict[str, Dict] = {}
    if not args.skip_table0_pure_ft:
        print("\n--- TABLE 0: Biography adaptation → re-probe gender (ΔE_bio) ---")
        rows_t0 = [
            ("B_task", model_b1),
            ("B_adv", model_b2),
            ("B_static_inlp", model_b3),
        ]
        if model_main is not None:
            rows_t0.append(("Main", model_main))
        for name, m in rows_t0:
            row = _biography_task_finetune_then_Ebio(
                m,
                train_ds,
                test_ds,
                device,
                args.batch_size,
                args.seed,
                ft_epochs=args.bio_ft_epochs,
                ft_lr=args.bio_ft_lr,
                tokenizer=tokenizer,
                adapt_objective=args.adapt_objective,
                adapt_lm_max_length=args.adapt_lm_max_length,
            )
            table0_pure_bio_task_ft[name] = row
            print(
                f"  {name}: E_before={row['E_bio_before_ft']} E_after={row['E_bio_after_bio_ft']} "
                f"ΔE={row['delta_E_bio_pure_ft']}"
            )

    meta = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "model": args.model,
        "seed": args.seed,
        "bios_train": len(train_ds),
        "bios_val": len(val_ds),
        "bios_test": len(test_ds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "lambda_bias": args.lambda_bias,
        "lambda_stab": args.lambda_stab,
        "main_inner_lr": args.main_inner_lr,
        "main_inner_steps": args.main_inner_steps,
        "main_stab_loss_mode": args.main_stab_loss_mode,
        "inlp_iterations": args.inlp_iterations,
        "inlp_iterations_used": inlp_k,
        "lambda_bias_train_used": train_lambda_bias,
        "weak_debias": args.weak_debias,
        "b2_lora": args.b2_lora,
        "b2_lora_r": args.b2_lora_r if args.b2_lora else None,
        "b2_lora_alpha": args.b2_lora_alpha if args.b2_lora else None,
        "b2_lora_dropout": args.b2_lora_dropout if args.b2_lora else None,
        "biography_probe": not args.skip_biography_probe,
        "adaptation_steps": args.adaptation_steps,
        "adaptation_lr": args.adaptation_lr,
        "adaptation_task_only": args.adaptation_task_only,
        "adaptation_adapt_lora": args.adaptation_adapt_lora,
        "adaptation_grad_clip": args.adaptation_grad_clip,
        "recompute_projection_each_step": args.recompute_projection_each_step,
        "adapt_objective": args.adapt_objective,
        "adapt_lm_max_length": args.adapt_lm_max_length,
        "core_baselines": ["B_task (L_task)", "B_adv (L_task + λ1 L_bias)", "Main (L_task + λ1 L_bias + λ2 L_stab)"],
        "supporting_baselines": ["B_static_inlp (INLP)", "A2_runtime_dynamic_proj"],
        "probe_protocol": "Same sklearn Pipeline(StandardScaler, LogisticRegression) and random_state=seed for R1/R2/R3; 80/20 stratified probe split; scaler fit on probe train only.",
        "notes": "Main: `run_main` → `checkpoints/main/` (or legacy `checkpoints/b4_stability/` when skip-train). Same `_agentic_eval_for_model` as B1/B2.",
        "metric_interpretation": [
            "If R1≈R3≈1.0 for all models, the linear probe is saturated; use excess recoverability E=(R-p*)/(1-p*) and trajectory_delta_excess_R, and/or ROC-AUC.",
            "INLP projection P is fit on biography-pooled pretrained reps; agent prompts change the input distribution, so gender may remain linearly recoverable (R stays high) unless you align training/eval prompts or refit P on agent trajectories.",
            "To show debiasing, first verify low R (or low E) on biography-only hidden states; then measure drift after LoRA / inner adaptation / multi-step prompts.",
            "Biography-only columns in results: debiasing is evaluated on the same tokenized bios as B1/B2/B3. Agentic columns use multi-step prompts (distribution shift).",
            "Use --weak-debias for an intentional partial-suppression regime (low λ1, few INLP iters) so E_bio stays in (0.2,0.8) and trajectories have headroom.",
            "B2 default (no --b2-lora): full backbone is trained with AdamW(model.parameters())—GRL updates representations. Agentic inner-loop: task head only (no LoRA).",
            "B2 with --b2-lora: training freezes base backbone; LoRA + heads train. With adapt_objective='occupation', inner loop uses LoRA+task head on agentic step-1 tokens; with 'summarize_lm', inner loop uses LM loss on raw bios (LoRA only).",
            "TABLE0 default: causal LM on 'Summarize this biography.' + bio + pseudo-summary; classification heads frozen; re-probe gender on same biography inputs.",
            "adapt_objective='summarize_lm': task-shift via generation-style objective; debiasing remains 28-way occupation. Gender probe unchanged.",
        ],
        "baselines_run": list(results.keys()),
        "table0_pure_bio_task_ft": not args.skip_table0_pure_ft,
        "bio_ft_epochs": args.bio_ft_epochs,
        "bio_ft_lr": args.bio_ft_lr,
    }
    report = {"meta": meta, "results": results, "table0_pure_bio_task_ft": table0_pure_bio_task_ft}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"agentic_report_{ts}.json")
    md_path = os.path.join(RESULTS_DIR, f"agentic_report_{ts}.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_text = render_agentic_report_markdown(
        meta,
        results,
        table0_pure_bio_task_ft,
        skip_biography_probe=args.skip_biography_probe,
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"\nAgentic report JSON: {json_path}")
    print(f"Agentic report MD:   {md_path}")


if __name__ == "__main__":
    main()

