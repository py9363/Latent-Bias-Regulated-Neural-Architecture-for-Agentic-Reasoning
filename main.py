"""
Experiment runner: Bias in Bios as primary training and probing dataset.
Usage:
  python main.py --baseline b1 [--lambda 0.5] [--lora_rank 8]
  python main.py --baseline b2 --lambda 0.5
  python main.py --baseline b3
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ensure_dirs,
    get_device,
    RESULTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_LAMBDA_BIAS,
)
from data.loaders import get_qwen_tokenizer, load_crows_pairs, load_bbq
from data.bias_in_bios import load_bias_in_bios
from baselines.b1_standard import run_b1, _collate_batch
from baselines.b2_adversarial import run_b2
from baselines.b3_inlp import run_b3
from evaluation.probe import run_probe
from evaluation.metrics import (
    compute_occupation_accuracy_and_gender_gap,
    get_backbone_for_lm,
    evaluate_crows_pairs_with_model,
    evaluate_bbq_with_model,
)

# Bias in Bios: 28 occupations
NUM_OCCUPATIONS = 28


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
    """Build a small dataset for LoRA adaptation: add labels (LM format, pad=-100)."""
    n = min(subset_size, len(dataset))
    subset = dataset.select(range(n))

    def add_labels(ex):
        ids = ex["input_ids"][:max_length]
        lab = [x if x != pad_token_id else -100 for x in ids]
        return {"labels": lab}

    subset = subset.map(add_labels, desc="labels for LoRA")
    return subset


def main():
    parser = argparse.ArgumentParser(description="Bias in Bios experiment runner")
    parser.add_argument("--baseline", type=str, choices=["b1", "b2", "b3"], required=True)
    parser.add_argument("--lambda", dest="lambda_bias", type=float, default=DEFAULT_LAMBDA_BIAS,
                        help="B2 adversarial lambda (L_task + lambda * L_bias)")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank for adaptation simulation")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--bios_lora_subset", type=int, default=2000, help="Bios samples for LoRA adaptation")
    parser.add_argument("--crows_max", type=int, default=None)
    parser.add_argument("--bbq_config", type=str, default="Age_disambig")
    parser.add_argument("--bbq_max", type=int, default=None)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--no_lora", action="store_true", help="Skip LoRA adaptation and delta_R")
    parser.add_argument("--no_external", action="store_true", help="Skip CrowS-Pairs and BBQ evaluation")
    parser.add_argument(
        "--no_bias_loss_balance",
        action="store_true",
        help="B2: disable per-batch L_task/L_bias scaling on λ1·L_bias",
    )
    args = parser.parse_args()

    ensure_dirs()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    tokenizer = get_qwen_tokenizer(args.model)
    print("Loading Bias in Bios (train/val/test)...")
    train_ds, val_ds, test_ds = load_bias_in_bios(
        tokenizer,
        max_length=args.max_length,
        use_predefined_splits=True,
    )
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

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

    if args.baseline == "b1":
        print("\n--- B1: Standard fine-tuning ---")
        model, ckpt_path, reps_path = run_b1(**common_kw)
    elif args.baseline == "b2":
        print("\n--- B2: Adversarial debiasing ---")
        common_kw["lambda_bias"] = args.lambda_bias
        model, ckpt_path, reps_path = run_b2(
            **common_kw, balance_bias_loss=not args.no_bias_loss_balance
        )
    else:
        print("\n--- B3: INLP ---")
        common_kw["num_bias_labels"] = 2
        model, ckpt_path, reps_path = run_b3(**common_kw)

    # Occupation accuracy and gender gap on test
    occ_metrics = compute_occupation_accuracy_and_gender_gap(
        model, test_ds, device=device, batch_size=args.batch_size, collate_fn=_collate_batch
    )
    occupation_accuracy = occ_metrics["occupation_accuracy"]
    gender_gap = occ_metrics["gender_gap"]

    # Recoverability R(θ) from saved representations
    if reps_path and os.path.isfile(reps_path):
        probe_result = _run_probe_on_reps(reps_path, test_size=0.2)
        recoverability = round(probe_result["accuracy"], 4)
    else:
        recoverability = None

    # LoRA adaptation and delta_R
    delta_recoverability = None
    if not args.no_lora and reps_path:
        try:
            from adaptation.lora_adaptation import run_lora_adaptation, extract_hidden_after_lora
            lora_ds = _bios_subset_for_lora(
                train_ds, tokenizer, args.max_length,
                subset_size=args.bios_lora_subset,
                pad_token_id=tokenizer.pad_token_id or 0,
            )
            adapted_model, lora_dir = run_lora_adaptation(
                base_model_name=args.model,
                adaptation_dataset=lora_ds,
                lora_r=args.lora_rank,
                epochs=2,
                batch_size=4,
                max_length=args.max_length,
                device=device,
            )
            # Extract hidden states from adapted model on train subset (for probe)
            lora_subset = train_ds.select(range(min(args.bios_lora_subset, len(train_ds))))
            h_adapted, s_adapted = extract_hidden_after_lora(
                adapted_model, tokenizer, lora_subset,
                batch_size=args.batch_size, device=device,
            )
            import numpy as np
            probe_adapted = run_probe(
                h_adapted.numpy() if hasattr(h_adapted, "numpy") else h_adapted.cpu().numpy(),
                s_adapted, test_size=0.2,
            )
            R_prime = probe_adapted["accuracy"]
            delta_recoverability = round(R_prime - recoverability, 4) if recoverability is not None else None
        except Exception as e:
            print(f"LoRA adaptation failed: {e}")
            delta_recoverability = None

    # External fairness: CrowS-Pairs and BBQ
    crows_pairs_bias_score = None
    bbq_accuracy_gap = None
    if not args.no_external:
        backbone = get_backbone_for_lm(model)
        crows_ds = load_crows_pairs(split="test")
        if args.crows_max is not None and len(crows_ds) > args.crows_max:
            crows_ds = crows_ds.select(range(args.crows_max))
        print("  Evaluating CrowS-Pairs...")
        crows_result = evaluate_crows_pairs_with_model(
            backbone, tokenizer, crows_ds, device=device, batch_size=8, max_length=128
        )
        crows_pairs_bias_score = round(crows_result["crows_pairs_bias_score"], 2)
        bbq_ds = load_bbq(config=args.bbq_config, split="test")
        if args.bbq_max is not None and len(bbq_ds) > args.bbq_max:
            bbq_ds = bbq_ds.select(range(args.bbq_max))
        print("  Evaluating BBQ...")
        bbq_result = evaluate_bbq_with_model(
            backbone, tokenizer, bbq_ds, device=device, batch_size=4, max_length=256
        )
        bbq_accuracy_gap = round(bbq_result["bbq_accuracy_gap"], 2)

    log = {
        "occupation_accuracy": round(occupation_accuracy, 4),
        "gender_gap": round(gender_gap, 4),
        "recoverability": recoverability,
        "delta_recoverability": delta_recoverability,
        "crows_pairs_bias_score": crows_pairs_bias_score,
        "bbq_accuracy_gap": bbq_accuracy_gap,
    }
    meta = {
        "timestamp": datetime.now().isoformat(),
        "baseline": args.baseline,
        "model": args.model,
        "lambda_bias": getattr(args, "lambda_bias", None),
        "lora_rank": args.lora_rank,
        "bios_train_size": len(train_ds),
        "bios_test_size": len(test_ds),
    }
    output = {"meta": meta, "log": log}
    print("\nResult log:", json.dumps(log, indent=2))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"bios_{args.baseline}_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
