"""
Run baselines B0–B3 and **Main** (always) on Bias in Bios + CrowS-Pairs + BBQ; write report.
B1=no suppression, B2=adversarial (static), B3=INLP. **Main** = stability-regularized adversarial (``run_main``, LoRA).

Typical capstone flows (from repo root, PowerShell):
  python run_agentic_baselines.py
      Trains B1, B2, Main, B3; agentic multi-step eval, biography probes, TABLE 0–5 → JSON/MD in RESULTS_DIR.

  python run_all_baselines.py
      B0–B3 + **Main** + CrowS/BBQ + LoRA ΔR (unless ``--no-lora``); JSON/MD in RESULTS_DIR.

See ``run_agentic_baselines.py`` docstring for CONFIG (adaptation steps, λ1/λ2, data caps).
Usage: python run_all_baselines.py [--lambda-bias F] [--lora-r R] [--lora-alpha A] [--no-bias-loss-balance] ...
"""
import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
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
from data.loaders import get_qwen_tokenizer, load_crows_pairs, load_bbq
from data.bias_in_bios import load_bias_in_bios
from baselines.b1_standard import run_b1, _collate_batch
from baselines.b2_adversarial import run_b2
from baselines.b3_inlp import run_b3
from baselines.main_stability import run_main
from models.qwen_task import QwenTaskModel
from evaluation.probe import run_probe
from evaluation.metrics import (
    compute_occupation_accuracy_and_gender_gap,
    get_backbone_for_lm,
    evaluate_crows_pairs_with_model,
    evaluate_bbq_with_model,
)

NUM_OCCUPATIONS = 28
DEFAULT_SEED = 42


def set_seed(seed: int):
    """Set global seeds for reproducibility (PyTorch, NumPy, Python random)."""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optional: fully deterministic CUDA (can slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_data(tokenizer, max_length=256, crows_max=None, bbq_max=None):
    train_ds, val_ds, test_ds = load_bias_in_bios(
        tokenizer, max_length=max_length, use_predefined_splits=True
    )
    crows = load_crows_pairs(split="test")
    if crows_max is not None and len(crows) > crows_max:
        crows = crows.select(range(crows_max))
    # Load all BBQ configs so protected_attribute varies → meaningful accuracy gap
    bbq = load_bbq(config=None, split="test")
    if bbq_max is not None and len(bbq) > bbq_max:
        bbq = bbq.select(range(bbq_max))
    return train_ds, val_ds, test_ds, crows, bbq


def _run_probe_on_reps(reps_path, **kwargs):
    import torch
    try:
        data = torch.load(reps_path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(reps_path, map_location="cpu")
    return run_probe(
        data["hidden_states"],
        data["sensitive_attributes"],
        **kwargs,
    )


def _bios_subset_for_lora(dataset, tokenizer, max_length: int, subset_size: int = 2000, pad_token_id: int = 0):
    """Subset of Bias in Bios with LM labels for LoRA adaptation."""
    n = min(subset_size, len(dataset))
    subset = dataset.select(range(n))

    def add_labels(ex):
        ids = ex["input_ids"][:max_length]
        lab = [x if x != pad_token_id else -100 for x in ids]
        return {"labels": lab}

    return subset.map(add_labels, desc="labels for LoRA")


def main():
    parser = argparse.ArgumentParser(description="Run all baselines and produce report")
    parser.add_argument("--crows-max", type=int, default=None, help="Max CrowS-Pairs examples (default: all)")
    parser.add_argument("--bbq-max", type=int, default=None, help="Max BBQ examples (default: all)")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length for Bias in Bios")
    parser.add_argument("--bios-train-max", type=int, default=10000, help="Cap Bias in Bios train size (default: 10000)")
    parser.add_argument("--bios-val-max", type=int, default=5000, help="Cap Bias in Bios val size (default: 5000)")
    parser.add_argument("--bios-test-max", type=int, default=10000, help="Cap Bias in Bios test size for eval (default: 10000)")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (default: config)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--no-lora", action="store_true", help="Skip LoRA adaptation and delta_R")
    parser.add_argument("--quick", action="store_true", help="Quick run: subset Bios train, 200 crows, 200 bbq, 2 epochs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--lambda-bias",
        type=float,
        default=None,
        help=f"Adversarial λ1 for B2 + Main (default: {DEFAULT_LAMBDA_BIAS}; try 2.0 for stronger pressure)",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help=f"LoRA rank for B2/Main (default: {DEFAULT_LORA_R})",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help=f"LoRA alpha for B2/Main (default: {DEFAULT_LORA_ALPHA})",
    )
    parser.add_argument(
        "--no-bias-loss-balance",
        action="store_true",
        help="Disable per-batch L_task/L_bias magnitude scaling on the adversarial term",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")

    if args.quick:
        if args.crows_max is None:
            args.crows_max = 200
        if args.bbq_max is None:
            args.bbq_max = 200
        if args.epochs is None:
            args.epochs = 2
    if args.epochs is None:
        args.epochs = DEFAULT_EPOCHS

    lambda_bias_train = args.lambda_bias if args.lambda_bias is not None else DEFAULT_LAMBDA_BIAS
    lora_r_train = args.lora_r if args.lora_r is not None else DEFAULT_LORA_R
    lora_alpha_train = args.lora_alpha if args.lora_alpha is not None else DEFAULT_LORA_ALPHA
    balance_bias_loss = not args.no_bias_loss_balance

    ensure_dirs()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    log_device_banner(device)

    tokenizer = get_qwen_tokenizer(args.model)
    print("Loading datasets (Bias in Bios + CrowS-Pairs + BBQ)...")
    train_ds, val_ds, test_ds, crows_ds, bbq_ds = _load_data(
        tokenizer,
        max_length=args.max_length,
        crows_max=args.crows_max,
        bbq_max=args.bbq_max,
    )
    if args.quick and len(train_ds) > 5000:
        train_ds = train_ds.select(range(5000))
    if args.quick:
        if len(val_ds) > 1000:
            val_ds = val_ds.select(range(1000))
        if len(test_ds) > 2000:
            test_ds = test_ds.select(range(2000))
    if args.bios_train_max is not None and len(train_ds) > args.bios_train_max:
        train_ds = train_ds.select(range(args.bios_train_max))
    if args.bios_val_max is not None and len(val_ds) > args.bios_val_max:
        val_ds = val_ds.select(range(args.bios_val_max))
    if args.bios_test_max is not None and len(test_ds) > args.bios_test_max:
        test_ds = test_ds.select(range(args.bios_test_max))
    if len(train_ds) > 20000:
        print(f"  Note: Training on {len(train_ds)} samples will take a long time. Use --bios-train-max 20000 or --quick for faster runs.")
    print(f"  Bias in Bios: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"  CrowS-Pairs: {len(crows_ds)} | BBQ: {len(bbq_ds)}")
    print(
        f"  B2/Main: λ1={lambda_bias_train} | LoRA r={lora_r_train} α={lora_alpha_train} | "
        f"bias_head=MLP | bias_loss_balance={balance_bias_loss}"
    )

    results = {}
    ckpt_main = None
    common_kw = dict(
        train_dataset=train_ds,
        eval_dataset=val_ds,
        model_name=args.model,
        num_labels=NUM_OCCUPATIONS,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=device,
        save_representations=True,
    )

    # ---- B0: Pretrained Qwen (no fine-tuning) ----
    print("\n--- B0: Pretrained Qwen (no training) ---")
    import torch
    from torch.utils.data import DataLoader
    from datasets import concatenate_datasets
    model_b0 = QwenTaskModel(model_name=args.model, num_labels=NUM_OCCUPATIONS)
    model_b0 = model_b0.to(device)
    model_b0.eval()
    # Extract hidden states for probe (use train subset to match other baselines)
    reps_dir_b0 = os.path.join("representations", "b0_pretrained")
    os.makedirs(reps_dir_b0, exist_ok=True)
    full_for_reps = concatenate_datasets([train_ds, val_ds]) if len(val_ds) > 0 else train_ds
    n_reps = min(len(full_for_reps), 5000)
    ds_reps = full_for_reps.select(range(n_reps))
    loader_reps = DataLoader(ds_reps, batch_size=args.batch_size, collate_fn=_collate_batch)
    all_hidden_b0, all_sens_b0 = [], []
    with torch.no_grad():
        for batch in loader_reps:
            out = model_b0(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                return_hidden=True,
            )
            all_hidden_b0.append(out["hidden_states"].cpu())
            all_sens_b0.extend(batch["sensitive_attribute"])
    hidden_b0 = torch.cat(all_hidden_b0, dim=0)
    reps_path_b0 = os.path.join(reps_dir_b0, "hidden_and_metadata.pt")
    torch.save({"hidden_states": hidden_b0, "sensitive_attributes": all_sens_b0}, reps_path_b0)
    probe_b0 = _run_probe_on_reps(reps_path_b0, test_size=0.2, random_state=args.seed)
    occ_b0 = compute_occupation_accuracy_and_gender_gap(
        model_b0, test_ds, device=device, batch_size=args.batch_size, collate_fn=_collate_batch
    )
    backbone_b0 = get_backbone_for_lm(model_b0)
    print("  Evaluating CrowS-Pairs...")
    crows_b0 = evaluate_crows_pairs_with_model(
        backbone_b0, tokenizer, crows_ds, device, batch_size=8, max_length=128
    )
    print("  Evaluating BBQ...")
    bbq_b0 = evaluate_bbq_with_model(
        backbone_b0, tokenizer, bbq_ds, device, batch_size=4, max_length=256
    )
    results["B0_pretrained"] = {
        "occupation_accuracy": round(occ_b0["occupation_accuracy"], 4),
        "gender_gap": round(occ_b0["gender_gap"], 4),
        "recoverability_R": round(probe_b0["accuracy"], 4),
        "probe_chance_baseline": round(probe_b0["chance_baseline"], 4),
        "probe_roc_auc": probe_b0.get("roc_auc"),
        "crows_pairs_bias_score": round(crows_b0["crows_pairs_bias_score"], 2),
        "bbq_task_accuracy": round(bbq_b0["task_accuracy"], 2),
        "bbq_accuracy_gap": round(bbq_b0["bbq_accuracy_gap"], 2),
    }
    del model_b0
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- B1 ----
    print("\n--- B1: Standard fine-tuning ---")
    model_b1, ckpt_b1, reps_b1 = run_b1(**common_kw)
    occ_b1 = compute_occupation_accuracy_and_gender_gap(
        model_b1, test_ds, device=device, batch_size=args.batch_size, collate_fn=_collate_batch
    )
    probe_b1 = _run_probe_on_reps(reps_b1, test_size=0.2, random_state=args.seed)
    backbone_b1 = get_backbone_for_lm(model_b1)
    print("  Evaluating CrowS-Pairs...")
    crows_b1 = evaluate_crows_pairs_with_model(
        backbone_b1, tokenizer, crows_ds, device, batch_size=8, max_length=128
    )
    print("  Evaluating BBQ...")
    bbq_b1 = evaluate_bbq_with_model(
        backbone_b1, tokenizer, bbq_ds, device, batch_size=4, max_length=256
    )
    results["B1_standard"] = {
        "occupation_accuracy": round(occ_b1["occupation_accuracy"], 4),
        "gender_gap": round(occ_b1["gender_gap"], 4),
        "recoverability_R": round(probe_b1["accuracy"], 4),
        "probe_chance_baseline": round(probe_b1["chance_baseline"], 4),
        "probe_roc_auc": probe_b1.get("roc_auc"),
        "crows_pairs_bias_score": round(crows_b1["crows_pairs_bias_score"], 2),
        "bbq_task_accuracy": round(bbq_b1["task_accuracy"], 2),
        "bbq_accuracy_gap": round(bbq_b1["bbq_accuracy_gap"], 2),
    }
    del model_b1
    import torch
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- B2 ----
    print("\n--- B2: Adversarial debiasing (LoRA + MLP bias head) ---")
    model_b2, ckpt_b2, reps_b2 = run_b2(
        **common_kw,
        lambda_bias=lambda_bias_train,
        use_lora=True,
        lora_r=lora_r_train,
        lora_alpha=lora_alpha_train,
        lora_dropout=DEFAULT_LORA_DROPOUT,
        balance_bias_loss=balance_bias_loss,
    )
    occ_b2 = compute_occupation_accuracy_and_gender_gap(
        model_b2, test_ds, device=device, batch_size=args.batch_size, collate_fn=_collate_batch
    )
    probe_b2 = _run_probe_on_reps(reps_b2, test_size=0.2, random_state=args.seed)
    backbone_b2 = get_backbone_for_lm(model_b2)
    print("  Evaluating CrowS-Pairs...")
    crows_b2 = evaluate_crows_pairs_with_model(
        backbone_b2, tokenizer, crows_ds, device, batch_size=8, max_length=128
    )
    print("  Evaluating BBQ...")
    bbq_b2 = evaluate_bbq_with_model(
        backbone_b2, tokenizer, bbq_ds, device, batch_size=4, max_length=256
    )
    results["B2_adversarial"] = {
        "occupation_accuracy": round(occ_b2["occupation_accuracy"], 4),
        "gender_gap": round(occ_b2["gender_gap"], 4),
        "recoverability_R": round(probe_b2["accuracy"], 4),
        "probe_chance_baseline": round(probe_b2["chance_baseline"], 4),
        "probe_roc_auc": probe_b2.get("roc_auc"),
        "crows_pairs_bias_score": round(crows_b2["crows_pairs_bias_score"], 2),
        "bbq_task_accuracy": round(bbq_b2["task_accuracy"], 2),
        "bbq_accuracy_gap": round(bbq_b2["bbq_accuracy_gap"], 2),
    }
    del model_b2
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- B3 ----
    print("\n--- B3: INLP ---")
    model_b3, ckpt_b3, reps_b3 = run_b3(**common_kw)
    if reps_b3 is None:
        reps_b3 = os.path.join("representations", "b3_inlp", "hidden_and_metadata.pt")
        if not os.path.isfile(reps_b3):
            probe_b3 = {"accuracy": 0.0, "chance_baseline": 0.5, "roc_auc": None}
        else:
            probe_b3 = _run_probe_on_reps(reps_b3, test_size=0.2, random_state=args.seed)
    else:
        probe_b3 = _run_probe_on_reps(reps_b3, test_size=0.2, random_state=args.seed)
    occ_b3 = compute_occupation_accuracy_and_gender_gap(
        model_b3, test_ds, device=device, batch_size=args.batch_size, collate_fn=_collate_batch
    )
    backbone_b3 = get_backbone_for_lm(model_b3)
    print("  Evaluating CrowS-Pairs...")
    crows_b3 = evaluate_crows_pairs_with_model(
        backbone_b3, tokenizer, crows_ds, device, batch_size=8, max_length=128
    )
    print("  Evaluating BBQ...")
    bbq_b3 = evaluate_bbq_with_model(
        backbone_b3, tokenizer, bbq_ds, device, batch_size=4, max_length=256
    )
    results["B3_INLP"] = {
        "occupation_accuracy": round(occ_b3["occupation_accuracy"], 4),
        "gender_gap": round(occ_b3["gender_gap"], 4),
        "recoverability_R": round(probe_b3["accuracy"], 4),
        "probe_chance_baseline": round(probe_b3["chance_baseline"], 4),
        "probe_roc_auc": probe_b3.get("roc_auc"),
        "crows_pairs_bias_score": round(crows_b3["crows_pairs_bias_score"], 2),
        "bbq_task_accuracy": round(bbq_b3["task_accuracy"], 2),
        "bbq_accuracy_gap": round(bbq_b3["bbq_accuracy_gap"], 2),
    }
    del model_b3
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- Main (stability-regularized adversarial; always) ----
    print("\n--- Main: Stability-regularized adversarial (LoRA) ---")
    model_main, ckpt_main, reps_main = run_main(
        train_dataset=train_ds,
        eval_dataset=val_ds,
        model_name=args.model,
        num_labels=NUM_OCCUPATIONS,
        num_bias_labels=2,
        lambda_bias=lambda_bias_train,
        lambda_stab=DEFAULT_LAMBDA_STAB,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=DEFAULT_LR,
        device=device,
        save_representations=True,
        use_lora=True,
        lora_r=lora_r_train,
        lora_alpha=lora_alpha_train,
        lora_dropout=DEFAULT_LORA_DROPOUT,
        inner_lr=DEFAULT_MAIN_INNER_LR,
        inner_steps=DEFAULT_MAIN_INNER_STEPS,
        balance_bias_loss=balance_bias_loss,
    )
    occ_m = compute_occupation_accuracy_and_gender_gap(
        model_main, test_ds, device=device, batch_size=args.batch_size, collate_fn=_collate_batch
    )
    probe_m = _run_probe_on_reps(reps_main, test_size=0.2, random_state=args.seed)
    backbone_main = get_backbone_for_lm(model_main)
    print("  Evaluating CrowS-Pairs (Main)...")
    crows_m = evaluate_crows_pairs_with_model(
        backbone_main, tokenizer, crows_ds, device, batch_size=8, max_length=128
    )
    print("  Evaluating BBQ (Main)...")
    bbq_m = evaluate_bbq_with_model(
        backbone_main, tokenizer, bbq_ds, device, batch_size=4, max_length=256
    )
    results["Main"] = {
        "occupation_accuracy": round(occ_m["occupation_accuracy"], 4),
        "gender_gap": round(occ_m["gender_gap"], 4),
        "recoverability_R": round(probe_m["accuracy"], 4),
        "probe_chance_baseline": round(probe_m["chance_baseline"], 4),
        "probe_roc_auc": probe_m.get("roc_auc"),
        "crows_pairs_bias_score": round(crows_m["crows_pairs_bias_score"], 2),
        "bbq_task_accuracy": round(bbq_m["task_accuracy"], 2),
        "bbq_accuracy_gap": round(bbq_m["bbq_accuracy_gap"], 2),
    }
    del model_main
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- LoRA from each baseline (optional): R(θ), R(θ'), ΔR per baseline ----
    if not args.no_lora:
        from adaptation.lora_adaptation import (
            run_lora_from_baseline_checkpoint,
            extract_hidden_after_lora,
        )
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        lora_adaptation_dataset = _bios_subset_for_lora(
            train_ds, tokenizer, args.max_length, subset_size=2000, pad_token_id=pad_id
        )
        baselines_for_lora = [
            ("B0_pretrained", None, "b0"),  # no checkpoint: use pretrained weights as-is
            ("B1_standard", ckpt_b1, "b1"),
            ("B2_adversarial", ckpt_b2, "b2"),
            ("B3_INLP", ckpt_b3, "b3"),
        ]
        if ckpt_main is not None:
            baselines_for_lora.append(("Main", ckpt_main, "main"))
        for key, ckpt_path, base_name in baselines_for_lora:
            if key not in results:
                continue
            print(f"\n--- LoRA from {key} ---")
            adapted_model = None
            try:
                adapted_model, _ = run_lora_from_baseline_checkpoint(
                    baseline_checkpoint_path=ckpt_path,
                    base_model_name=args.model,
                    adaptation_dataset=lora_adaptation_dataset,
                    baseline_name=base_name,
                    epochs=2,
                    lr=1e-4,
                    batch_size=4,
                    max_length=args.max_length,
                    device=device,
                )
                h_adapted, s_adapted = extract_hidden_after_lora(
                    adapted_model, tokenizer, train_ds,
                    batch_size=args.batch_size, device=device,
                )
                probe_adapted = run_probe(
                    h_adapted.numpy() if hasattr(h_adapted, "numpy") else h_adapted.cpu().numpy(),
                    s_adapted, test_size=0.2, random_state=args.seed,
                )
                R_theta_prime = round(probe_adapted["accuracy"], 4)
                R_theta = results[key]["recoverability_R"]
                delta_R = round(R_theta_prime - R_theta, 4)
                results[key]["R_theta_prime"] = R_theta_prime
                results[key]["delta_R"] = delta_R
                print(f"  {key}: R(θ)={R_theta} → R(θ')={R_theta_prime}, ΔR={delta_R}")
            except Exception as e:
                print(f"  LoRA from {key} failed: {e}")
                results[key]["R_theta_prime"] = None
                results[key]["delta_R"] = None
            if adapted_model is not None:
                del adapted_model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    # ---- Report ----
    meta = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "model": args.model,
        "seed": args.seed,
        "baselines_run": ["B0_pretrained", "B1_standard", "B2_adversarial", "B3_INLP", "Main"],
        "bios_train": len(train_ds),
        "bios_val": len(val_ds),
        "bios_test": len(test_ds),
        "crows_examples": len(crows_ds),
        "bbq_examples": len(bbq_ds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lambda_bias": lambda_bias_train,
        "lora_r": lora_r_train,
        "lora_alpha": lora_alpha_train,
        "bias_head": "mlp",
        "bias_loss_balance": balance_bias_loss,
    }
    report = {"meta": meta, "results": results}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"report_{ts}.json")
    md_path = os.path.join(RESULTS_DIR, f"report_{ts}.md")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport JSON: {json_path}")

    # Markdown table
    lines = [
        "# Baseline Report (Bias in Bios)",
        "",
        f"**Generated:** {meta['timestamp']}",
        f"**Model:** {meta['model']} | **Device:** {meta['device']}",
        f"**Data:** Bios train={meta['bios_train']} val={meta['bios_val']} test={meta['bios_test']} | CrowS={meta['crows_examples']} | BBQ={meta['bbq_examples']}",
        "",
        "| Baseline | Occupation Acc % | Gender gap % | R(θ) | R(θ') | ΔR | CrowS-Pairs bias % | BBQ Acc | BBQ gap |",
        "|----------|------------------|---------------|------|-------|-----|---------------------|---------|---------|",
    ]
    for name, r in results.items():
        occ_acc = r.get("occupation_accuracy", "")
        gender_gap = r.get("gender_gap", "")
        R = r.get("recoverability_R", "")
        R_prime = r.get("R_theta_prime", "")
        delta_R = r.get("delta_R", "")
        crows = r.get("crows_pairs_bias_score", "")
        bbq_acc = r.get("bbq_task_accuracy", "")
        bbq_gap = r.get("bbq_accuracy_gap", "")
        lines.append(f"| {name} | {occ_acc} | {gender_gap} | {R} | {R_prime} | {delta_R} | {crows} | {bbq_acc} | {bbq_gap} |")
    lines.append("")
    lines.append("R(θ) = recoverability (probe accuracy on gender); R(θ') = after LoRA adaptation from that baseline; ΔR = R(θ') − R(θ).")
    lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report MD:   {md_path}")

    # Update baseline_comparison.json for easy reference (R(θ), R(θ'), ΔR per baseline)
    comparison = {
        "description": "Baseline comparison (Bias in Bios). R(θ)=recoverability; R(θ')=after LoRA from that baseline; ΔR=R(θ')−R(θ).",
        "generated": meta["timestamp"],
        "model": meta["model"],
        "device": meta["device"],
        "data": {
            "bios_train": meta["bios_train"],
            "bios_val": meta["bios_val"],
            "bios_test": meta["bios_test"],
            "crows_pairs": meta["crows_examples"],
            "bbq": meta["bbq_examples"],
        },
        "baselines": {},
        "summary": "",
    }
    for name, r in results.items():
        comparison["baselines"][name] = {
            "name": name.replace("_", " ").title(),
            "occupation_accuracy_percent": r.get("occupation_accuracy"),
            "gender_gap_percent": r.get("gender_gap"),
            "R_theta": r.get("recoverability_R"),
            "R_theta_prime": r.get("R_theta_prime"),
            "delta_R": r.get("delta_R"),
            "crows_pairs_bias_percent": r.get("crows_pairs_bias_score"),
            "bbq_accuracy_percent": r.get("bbq_task_accuracy"),
            "bbq_gap_percent": r.get("bbq_accuracy_gap"),
        }
    comparison_path = os.path.join(RESULTS_DIR, "baseline_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Updated: {comparison_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
