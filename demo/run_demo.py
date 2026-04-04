#!/usr/bin/env python3
"""
Bias in Bios **full baseline demo** on a **tiny shared split** (same protocol as ``run_all_baselines.py``,
without CrowS-Pairs / BBQ).

Runs in order:
  **B0** pretrained probe + task metrics on test  
  **B1** standard fine-tune  
  **B2** adversarial + LoRA (``run_b2``)  
  **B3** INLP (``run_b3``, few iterations)  
  **Main** stability-regularized adversarial (``run_main``)

Checkpoints: ``demo/output/checkpoints/{b1,b2,b3,main}/``  
Log / table: ``demo/output/demo_log.txt``, ``demo/output/demo_full_report.json``  
Agentic paper tables (same protocol as ``run_agentic_baselines.py``): ``demo/output/demo_agentic_report.md`` + ``demo_agentic_report.json``

Skip agentic eval (faster): ``--skip-agentic-report``. Skip TABLE 0 pure FT (memory): ``--skip-table0``.

From repo root:
  python demo/run_demo.py

Smaller / faster:
  python demo/run_demo.py --train-samples 48 --val-samples 12 --test-samples 24 --epochs 1 --inlp-k 2 --batch-size 4

Skip Main (e.g. very slow on CPU inner loop):
  python demo/run_demo.py --skip-main

Default debiasing uses milder λ and capped INLP k so tiny splits do not collapse task accuracy;
use ``--full-debias`` for ``config.DEFAULT_*`` strengths (often ~0% occupation acc on small n with few epochs).
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from datasets import concatenate_datasets
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.b1_standard import _collate_batch, run_b1  # noqa: E402
from baselines.b2_adversarial import load_b2_from_checkpoint, run_b2  # noqa: E402
from baselines.b3_inlp import run_b3  # noqa: E402
from baselines.main_stability import load_main_from_checkpoint, run_main  # noqa: E402
from config import (  # noqa: E402
    DEFAULT_ADAPTATION_LR,
    DEFAULT_ADAPTATION_STEPS,
    DEFAULT_LAMBDA_BIAS,
    DEFAULT_LAMBDA_STAB,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_LR,
    DEFAULT_MAIN_INNER_LR,
    DEFAULT_MAIN_INNER_STEPS,
    get_device,
    log_device_banner,
)
from data.adaptation_labels import (  # noqa: E402
    ADAPT_OBJECTIVE_OCCUPATION,
    ADAPT_OBJECTIVE_SUMMARIZE_LM,
)
from data.bias_in_bios import NUM_OCCUPATIONS, load_bias_in_bios  # noqa: E402
from data.loaders import get_qwen_tokenizer  # noqa: E402
from evaluation.agentic_report_md import render_agentic_report_markdown  # noqa: E402
from evaluation.metrics import compute_occupation_accuracy_and_gender_gap  # noqa: E402
from evaluation.probe import run_probe  # noqa: E402
from run_agentic_baselines import (  # noqa: E402
    _agentic_eval_for_model,
    _biography_probe_dict,
    _biography_task_finetune_then_Ebio,
    _merge_biography_and_lift,
)
from models.adversarial import QwenAdversarialModel  # noqa: E402
from models.qwen_task import QwenTaskModel  # noqa: E402

# Tiny-split defaults: full config λ1/λ2 and INLP k are tuned for larger data; here they collapse task
# accuracy in a single epoch. Milder demo weights keep B2/B3/Main readable without changing full runs.
# Milder than DEFAULT_LAMBDA_BIAS (1.0) for tiny n; use --full-debias for config defaults.
DEMO_LAMBDA_BIAS = 0.45
DEMO_LAMBDA_STAB = 0.04
DEMO_MAIN_INNER_STEPS = 4


def _demo_inlp_k(requested: int, n_train: int) -> int:
    """Cap INLP iterations vs train size (high-dim h + small n → noisy nullspace)."""
    cap = max(1, n_train // 64)
    return min(int(requested), cap)


def _load_b1_checkpoint(path: Path, model_name: str, device: str, num_labels: int) -> QwenTaskModel:
    m = QwenTaskModel(model_name=model_name, num_labels=num_labels).to(device)
    state = torch.load(path, map_location="cpu", weights_only=False)
    m.load_state_dict(state["model_state_dict"], strict=False)
    return m


def _load_b3_checkpoint(path: Path, model_name: str, device: str, num_labels: int) -> QwenTaskModel:
    state = torch.load(path, map_location="cpu", weights_only=False)
    proj = state.get("projection_matrix")
    m = QwenTaskModel(model_name=model_name, num_labels=num_labels, projection_matrix=proj).to(device)
    m.load_state_dict(state["model_state_dict"], strict=False)
    return m


def _demo_agentic_eval_and_write_reports(
    *,
    train_ds,
    val_ds,
    test_ds,
    tokenizer,
    device: str,
    args: Any,
    ckpt_root: Path,
    out_dir: Path,
    lambda_bias: float,
    lambda_stab: float,
    inlp_k: int,
    skip_table0: bool,
    bio_ft_epochs: int,
    main_inner_steps: int,
    adapt_objective: str,
    adapt_lm_max_length: int,
) -> Dict[str, Any]:
    """
    Same agentic metrics as ``run_agentic_baselines.py`` (TABLE 0–5). Loads one checkpoint at a time.
    Writes ``demo_agentic_report.md`` and ``demo_agentic_report.json`` under ``out_dir``.
    """
    eval_kw = dict(
        tokenizer=tokenizer,
        dataset=test_ds,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        seed=args.seed,
        adaptation_steps=DEFAULT_ADAPTATION_STEPS,
        adaptation_lr=DEFAULT_ADAPTATION_LR,
        recompute_projection_each_step=False,
        lambda_bias_adapt=lambda_bias,
        adaptation_task_only=True,
        adaptation_adapt_lora=True,
        adaptation_grad_clip=1.0,
        adapt_objective=adapt_objective,
        adapt_lm_max_length=adapt_lm_max_length,
    )
    agentic_results: Dict[str, Dict[str, Any]] = {}
    table0: Dict[str, Dict[str, float]] = {}

    print("\n--- Agentic evaluation (TABLE 0–5; test split, multi-step prompts) ---")

    b1_pt = ckpt_root / "b1_demo" / "pytorch_model.pt"
    m1 = _load_b1_checkpoint(b1_pt, args.model, device, NUM_OCCUPATIONS)
    agentic_results["B_task"] = _agentic_eval_for_model(model=m1, apply_dynamic_reg=False, **eval_kw)
    _merge_biography_and_lift(
        agentic_results["B_task"],
        _biography_probe_dict(m1, test_ds, args.batch_size, device, args.seed),
    )
    agentic_results["A2_runtime_dynamic_proj"] = _agentic_eval_for_model(
        model=m1, apply_dynamic_reg=True, **eval_kw
    )
    _merge_biography_and_lift(
        agentic_results["A2_runtime_dynamic_proj"],
        _biography_probe_dict(m1, test_ds, args.batch_size, device, args.seed),
    )
    if not skip_table0:
        table0["B_task"] = _biography_task_finetune_then_Ebio(
            m1,
            train_ds,
            test_ds,
            device,
            args.batch_size,
            args.seed,
            ft_epochs=bio_ft_epochs,
            ft_lr=5e-5,
            tokenizer=tokenizer,
            adapt_objective=adapt_objective,
            adapt_lm_max_length=adapt_lm_max_length,
        )
    del m1
    _free_cuda()

    b2_pt = ckpt_root / "b2_demo" / "pytorch_model.pt"
    m2 = load_b2_from_checkpoint(str(b2_pt), args.model, device, NUM_OCCUPATIONS, num_bias_labels=2)
    agentic_results["B_adv"] = _agentic_eval_for_model(model=m2, apply_dynamic_reg=False, **eval_kw)
    _merge_biography_and_lift(
        agentic_results["B_adv"],
        _biography_probe_dict(m2, test_ds, args.batch_size, device, args.seed),
    )
    if not skip_table0:
        table0["B_adv"] = _biography_task_finetune_then_Ebio(
            m2,
            train_ds,
            test_ds,
            device,
            args.batch_size,
            args.seed,
            ft_epochs=bio_ft_epochs,
            ft_lr=5e-5,
            tokenizer=tokenizer,
            adapt_objective=adapt_objective,
            adapt_lm_max_length=adapt_lm_max_length,
        )
    del m2
    _free_cuda()

    b3_pt = ckpt_root / "b3_demo" / "pytorch_model.pt"
    m3 = _load_b3_checkpoint(b3_pt, args.model, device, NUM_OCCUPATIONS)
    agentic_results["B_static_inlp"] = _agentic_eval_for_model(model=m3, apply_dynamic_reg=False, **eval_kw)
    _merge_biography_and_lift(
        agentic_results["B_static_inlp"],
        _biography_probe_dict(m3, test_ds, args.batch_size, device, args.seed),
    )
    if not skip_table0:
        table0["B_static_inlp"] = _biography_task_finetune_then_Ebio(
            m3,
            train_ds,
            test_ds,
            device,
            args.batch_size,
            args.seed,
            ft_epochs=bio_ft_epochs,
            ft_lr=5e-5,
            tokenizer=tokenizer,
            adapt_objective=adapt_objective,
            adapt_lm_max_length=adapt_lm_max_length,
        )
    del m3
    _free_cuda()

    main_pt = ckpt_root / "main_demo" / "pytorch_model.pt"
    if not args.skip_main and main_pt.is_file():
        mm = load_main_from_checkpoint(str(main_pt), args.model, device, NUM_OCCUPATIONS, num_bias_labels=2)
        agentic_results["Main"] = _agentic_eval_for_model(model=mm, apply_dynamic_reg=False, **eval_kw)
        _merge_biography_and_lift(
            agentic_results["Main"],
            _biography_probe_dict(mm, test_ds, args.batch_size, device, args.seed),
        )
        if not skip_table0:
            table0["Main"] = _biography_task_finetune_then_Ebio(
                mm,
                train_ds,
                test_ds,
                device,
                args.batch_size,
                args.seed,
                ft_epochs=bio_ft_epochs,
                ft_lr=5e-5,
                tokenizer=tokenizer,
                adapt_objective=adapt_objective,
                adapt_lm_max_length=adapt_lm_max_length,
            )
        del mm
        _free_cuda()

    agentic_meta = {
        "timestamp": datetime.now().isoformat(),
        "report_kind": "demo_tiny_split",
        "device": device,
        "model": args.model,
        "seed": args.seed,
        "bios_train": len(train_ds),
        "bios_val": len(val_ds),
        "bios_test": len(test_ds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "lambda_bias": lambda_bias,
        "lambda_bias_train_used": lambda_bias,
        "lambda_stab": lambda_stab,
        "inlp_iterations": args.inlp_k,
        "inlp_iterations_used": inlp_k,
        "main_inner_steps_used": main_inner_steps,
        "full_debias": args.full_debias,
        "skip_main_training": args.skip_main,
        "skip_table0": skip_table0,
        "bio_ft_epochs": 0 if skip_table0 else bio_ft_epochs,
        "bio_ft_lr": 5e-5,
        "adaptation_steps": DEFAULT_ADAPTATION_STEPS,
        "adaptation_lr": DEFAULT_ADAPTATION_LR,
        "adapt_objective": adapt_objective,
        "adapt_lm_max_length": adapt_lm_max_length,
        "probe_protocol": (
            "Same sklearn Pipeline(StandardScaler, LogisticRegression) and random_state=seed for R1/R2/R3; "
            "80/20 stratified probe split; scaler fit on probe train only."
        ),
        "baselines_run": list(agentic_results.keys()),
    }

    report = {
        "meta": agentic_meta,
        "results": agentic_results,
        "table0_pure_bio_task_ft": table0,
    }
    md_path = out_dir / "demo_agentic_report.md"
    json_path = out_dir / "demo_agentic_report.json"
    md_body = render_agentic_report_markdown(
        agentic_meta,
        agentic_results,
        table0,
        skip_biography_probe=False,
        report_title="Agentic Baseline Report (Demo — tiny split)",
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_body)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote agentic Markdown → {md_path}")
    print(f"Wrote agentic JSON    → {json_path}")
    return {"paths": {"markdown": str(md_path), "json": str(json_path)}, "report": report}


class _Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_sample_batch_json(tokenizer, dataset, path: Path, n: int = 4) -> None:
    rows = []
    for i in range(min(n, len(dataset))):
        ex = dataset[i]
        text = tokenizer.decode(ex["input_ids"], skip_special_tokens=True)
        rows.append(
            {
                "index": i,
                "text_preview": text[:400] + ("…" if len(text) > 400 else ""),
                "occupation_label": int(ex["label"]),
                "gender_label": int(ex["sensitive_attribute"]),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Wrote sample batch preview → {path}")


def _collect_hidden_unified(model, dataset, device: str, batch_size: int) -> tuple:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=_collate_batch, shuffle=False)
    h_list, s_list = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            if isinstance(model, QwenAdversarialModel):
                out = model(input_ids=ids, attention_mask=mask)
            else:
                out = model(input_ids=ids, attention_mask=mask, return_hidden=True)
            h_list.append(out["hidden_states"].cpu().numpy())
            s_list.extend(batch["sensitive_attribute"])
    return np.concatenate(h_list, axis=0), np.asarray(s_list, dtype=np.int64)


def _bios_metrics_row(
    model,
    train_ds,
    val_ds,
    test_ds,
    device: str,
    batch_size: int,
    seed: int,
) -> Dict[str, Any]:
    occ = compute_occupation_accuracy_and_gender_gap(
        model, test_ds, device=device, batch_size=batch_size, collate_fn=_collate_batch
    )
    if val_ds is not None and len(val_ds) > 0:
        probe_ds = concatenate_datasets([train_ds, val_ds])
    else:
        probe_ds = train_ds
    H, s = _collect_hidden_unified(model, probe_ds, device, batch_size)
    n = len(probe_ds)
    test_size = 0.2 if n >= 40 else max(0.15, min(0.35, 8 / max(n, 1)))
    probe = run_probe(H, s, test_size=test_size, random_state=seed)
    roc = probe.get("roc_auc")
    return {
        "occupation_accuracy": round(float(occ["occupation_accuracy"]), 4),
        "gender_gap": round(float(occ["gender_gap"]), 4),
        "recoverability_R": round(float(probe["accuracy"]), 4),
        "probe_chance_baseline": round(float(probe["chance_baseline"]), 4),
        "probe_roc_auc": None if roc is None else round(float(roc), 4),
    }


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _print_results_table(results: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "=" * 72)
    print("SUMMARY (Bias in Bios demo split)")
    print("=" * 72)
    print(
        f"{'Model':<18} {'Occ acc %':>10} {'Gender gap':>11} {'R_probe':>9} {'R_chance':>9}"
    )
    print("-" * 72)
    for name in ("B0_pretrained", "B1_standard", "B2_adversarial", "B3_INLP", "Main"):
        if name not in results:
            continue
        r = results[name]
        print(
            f"{name:<18} {r['occupation_accuracy']:>10.2f} {r['gender_gap']:>11.2f} "
            f"{r['recoverability_R']:>9.4f} {r['probe_chance_baseline']:>9.4f}"
        )
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run B0–B3 + Main on a tiny Bias in Bios split (demo)."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=96)
    parser.add_argument("--val-samples", type=int, default=24)
    parser.add_argument("--test-samples", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inlp-k", type=int, default=3, help="INLP iterations (capped vs train size unless --full-debias)")
    parser.add_argument(
        "--full-debias",
        action="store_true",
        help="Use config DEFAULT_LAMBDA_* / DEFAULT_MAIN_INNER_STEPS and raw --inlp-k (harsher; tiny data may collapse task acc).",
    )
    parser.add_argument("--skip-main", action="store_true", help="Skip Main (slow; uses CPU inner twin)")
    parser.add_argument(
        "--skip-agentic-report",
        action="store_true",
        help="Skip agentic TABLE 0–5 eval (no demo_agentic_report.md / .json)",
    )
    parser.add_argument(
        "--skip-table0",
        action="store_true",
        help="Skip TABLE 0 pure biography fine-tune + re-probe (saves memory)",
    )
    parser.add_argument(
        "--agentic-bio-ft-epochs",
        type=int,
        default=None,
        help="TABLE 0: extra L_task epochs on train bios (default 2 if n_train<200 else 3)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU for all models")
    parser.add_argument(
        "--no-bias-loss-balance",
        action="store_true",
        help="Disable L_task/L_bias magnitude scaling on adversarial term (B2/Main)",
    )
    parser.add_argument(
        "--adapt-objective",
        type=str,
        default=ADAPT_OBJECTIVE_SUMMARIZE_LM,
        choices=[ADAPT_OBJECTIVE_SUMMARIZE_LM, ADAPT_OBJECTIVE_OCCUPATION],
        help="TABLE 0 + agentic inner: LM 'Summarize this biography.' (task shift) vs 28-way occupation",
    )
    parser.add_argument(
        "--adapt-lm-max-length",
        type=int,
        default=512,
        help="Max tokens for LM summarize sequences (TABLE 0 + inner loop)",
    )
    args = parser.parse_args()

    demo_dir = Path(__file__).resolve().parent
    out_dir = demo_dir / "output"
    data_dir = demo_dir / "data"
    ckpt_root = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "demo_log.txt"
    report_path = out_dir / "demo_full_report.json"
    tee = _Tee(sys.stdout, open(log_path, "w", encoding="utf-8"))
    old_stdout = sys.stdout
    sys.stdout = tee

    bios_results: Dict[str, Dict[str, Any]] = {}
    meta: Dict[str, Any] = {}

    try:
        print("=" * 72)
        print("Bias in Bios — FULL BASELINE DEMO (B0, B1, B2, B3, Main)")
        print(f"UTC: {datetime.now(timezone.utc).isoformat()}")
        print("=" * 72)

        _set_seed(args.seed)
        device = "cpu" if args.cpu else get_device()
        log_device_banner(device)

        tokenizer = get_qwen_tokenizer(args.model)
        train_full, val_full, test_full = load_bias_in_bios(
            tokenizer, max_length=args.max_length, use_predefined_splits=True
        )

        n_tr = min(args.train_samples, len(train_full))
        n_va = min(args.val_samples, len(val_full))
        n_te = min(args.test_samples, len(test_full))
        train_ds = train_full.select(range(n_tr))
        val_ds = val_full.select(range(n_va))
        test_ds = test_full.select(range(n_te))

        print(f"\nDemo split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
        print(f"epochs={args.epochs} batch_size={args.batch_size} model={args.model}")

        if args.full_debias:
            lambda_bias = DEFAULT_LAMBDA_BIAS
            lambda_stab = DEFAULT_LAMBDA_STAB
            main_inner_steps = DEFAULT_MAIN_INNER_STEPS
            inlp_k = int(args.inlp_k)
            print("Debias: full config (DEFAULT_LAMBDA_*, DEFAULT_MAIN_INNER_STEPS, raw --inlp-k).")
        else:
            lambda_bias = DEMO_LAMBDA_BIAS
            lambda_stab = DEMO_LAMBDA_STAB
            main_inner_steps = DEMO_MAIN_INNER_STEPS
            inlp_k = _demo_inlp_k(args.inlp_k, len(train_ds))
            if inlp_k != args.inlp_k:
                print(
                    f"INLP: using k={inlp_k} (capped from --inlp-k={args.inlp_k} for n_train={len(train_ds)}). "
                    "Use --full-debias to force requested k."
                )
            print(
                f"Debias (demo-friendly): lambda_bias={lambda_bias} lambda_stab={lambda_stab} "
                f"main_inner_steps={main_inner_steps} | --full-debias for paper-style weights."
            )

        _write_sample_batch_json(tokenizer, train_ds, data_dir / "sample_batch.json", n=4)

        b3_epochs = args.epochs
        if not args.full_debias and len(train_ds) < 200:
            b3_epochs = max(args.epochs, 3)

        meta.update(
            {
                "model": args.model,
                "device": device,
                "train_n": len(train_ds),
                "val_n": len(val_ds),
                "test_n": len(test_ds),
                "epochs": args.epochs,
                "b3_epochs": b3_epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "inlp_k_requested": args.inlp_k,
                "inlp_k_used": inlp_k,
                "lambda_bias": lambda_bias,
                "lambda_stab": lambda_stab,
                "main_inner_steps": main_inner_steps,
                "full_debias": args.full_debias,
                "skip_main": args.skip_main,
                "skip_agentic_report": args.skip_agentic_report,
                "skip_table0": args.skip_table0,
                "agentic_bio_ft_epochs": args.agentic_bio_ft_epochs,
                "bias_loss_balance": not args.no_bias_loss_balance,
                "adapt_objective": args.adapt_objective,
                "adapt_lm_max_length": args.adapt_lm_max_length,
            }
        )

        # ---- B0: pretrained (no training) ----
        print("\n--- B0: Pretrained Qwen (no fine-tuning) ---")
        model_b0 = QwenTaskModel(model_name=args.model, num_labels=NUM_OCCUPATIONS).to(device)
        model_b0.eval()
        bios_results["B0_pretrained"] = _bios_metrics_row(
            model_b0, train_ds, val_ds, test_ds, device, args.batch_size, args.seed
        )
        print(
            f"  occ_acc={bios_results['B0_pretrained']['occupation_accuracy']}%  "
            f"R={bios_results['B0_pretrained']['recoverability_R']}"
        )
        del model_b0
        _free_cuda()

        # ---- B1 ----
        print("\n--- B1: Standard fine-tuning ---")
        model_b1, _, _ = run_b1(
            train_dataset=train_ds,
            eval_dataset=val_ds,
            model_name=args.model,
            num_labels=NUM_OCCUPATIONS,
            output_dir=str(ckpt_root / "b1_demo"),
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=DEFAULT_LR,
            device=device,
            save_representations=False,
        )
        bios_results["B1_standard"] = _bios_metrics_row(
            model_b1, train_ds, val_ds, test_ds, device, args.batch_size, args.seed
        )
        print(
            f"  occ_acc={bios_results['B1_standard']['occupation_accuracy']}%  "
            f"R={bios_results['B1_standard']['recoverability_R']}"
        )
        del model_b1
        _free_cuda()

        # ---- B2 adversarial + LoRA ----
        print("\n--- B2: Adversarial debiasing (LoRA) ---")
        model_b2, _, _ = run_b2(
            train_dataset=train_ds,
            eval_dataset=val_ds,
            model_name=args.model,
            num_labels=NUM_OCCUPATIONS,
            num_bias_labels=2,
            lambda_bias=lambda_bias,
            output_dir=str(ckpt_root / "b2_demo"),
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=DEFAULT_LR,
            device=device,
            save_representations=False,
            use_lora=True,
            lora_r=DEFAULT_LORA_R,
            lora_alpha=DEFAULT_LORA_ALPHA,
            lora_dropout=DEFAULT_LORA_DROPOUT,
            balance_bias_loss=not args.no_bias_loss_balance,
        )
        bios_results["B2_adversarial"] = _bios_metrics_row(
            model_b2, train_ds, val_ds, test_ds, device, args.batch_size, args.seed
        )
        print(
            f"  occ_acc={bios_results['B2_adversarial']['occupation_accuracy']}%  "
            f"R={bios_results['B2_adversarial']['recoverability_R']}"
        )
        del model_b2
        _free_cuda()

        # ---- B3 INLP ----
        print(f"\n--- B3: INLP (k={inlp_k}) ---")
        model_b3, _, _ = run_b3(
            train_dataset=train_ds,
            eval_dataset=val_ds,
            model_name=args.model,
            num_labels=NUM_OCCUPATIONS,
            k_iterations=inlp_k,
            output_dir=str(ckpt_root / "b3_demo"),
            batch_size=args.batch_size,
            epochs=b3_epochs,
            lr=DEFAULT_LR,
            device=device,
            save_representations=False,
        )
        bios_results["B3_INLP"] = _bios_metrics_row(
            model_b3, train_ds, val_ds, test_ds, device, args.batch_size, args.seed
        )
        print(
            f"  occ_acc={bios_results['B3_INLP']['occupation_accuracy']}%  "
            f"R={bios_results['B3_INLP']['recoverability_R']}"
        )
        del model_b3
        _free_cuda()

        # ---- Main ----
        if not args.skip_main:
            print("\n--- Main: Stability-regularized adversarial (LoRA) ---")
            model_m, _, _ = run_main(
                train_dataset=train_ds,
                eval_dataset=val_ds,
                model_name=args.model,
                num_labels=NUM_OCCUPATIONS,
                num_bias_labels=2,
                lambda_bias=lambda_bias,
                lambda_stab=lambda_stab,
                output_dir=str(ckpt_root / "main_demo"),
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=DEFAULT_LR,
                device=device,
                save_representations=False,
                use_lora=True,
                lora_r=DEFAULT_LORA_R,
                lora_alpha=DEFAULT_LORA_ALPHA,
                lora_dropout=DEFAULT_LORA_DROPOUT,
                inner_lr=DEFAULT_MAIN_INNER_LR,
                inner_steps=main_inner_steps,
                balance_bias_loss=not args.no_bias_loss_balance,
            )
            bios_results["Main"] = _bios_metrics_row(
                model_m, train_ds, val_ds, test_ds, device, args.batch_size, args.seed
            )
            print(
                f"  occ_acc={bios_results['Main']['occupation_accuracy']}%  "
                f"R={bios_results['Main']['recoverability_R']}"
            )
            del model_m
            _free_cuda()
        else:
            print("\n--- Main: skipped (--skip-main) ---")

        bio_ft_epochs = (
            args.agentic_bio_ft_epochs
            if args.agentic_bio_ft_epochs is not None
            else (2 if len(train_ds) < 200 else 3)
        )
        agentic_bundle: Optional[Dict[str, Any]] = None
        if not args.skip_agentic_report:
            agentic_bundle = _demo_agentic_eval_and_write_reports(
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                tokenizer=tokenizer,
                device=device,
                args=args,
                ckpt_root=ckpt_root,
                out_dir=out_dir,
                lambda_bias=lambda_bias,
                lambda_stab=lambda_stab,
                inlp_k=inlp_k,
                skip_table0=args.skip_table0,
                bio_ft_epochs=bio_ft_epochs,
                main_inner_steps=main_inner_steps,
                adapt_objective=args.adapt_objective,
                adapt_lm_max_length=args.adapt_lm_max_length,
            )
        else:
            print("\n--- Agentic report: skipped (--skip-agentic-report) ---")

        _print_results_table(bios_results)

        out_payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "meta": meta,
            "results": bios_results,
            "agentic_report": None if agentic_bundle is None else agentic_bundle["report"],
            "agentic_paths": None if agentic_bundle is None else agentic_bundle["paths"],
            "checkpoints_dir": str(ckpt_root),
            "log_path": str(log_path),
            "sample_batch_path": str(data_dir / "sample_batch.json"),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2)
        print(f"\nWrote JSON report → {report_path}")
        if agentic_bundle is not None:
            print(f"Agentic report (TABLE 0–5) → {agentic_bundle['paths']['markdown']}")
        print(f"Full log → {log_path}")
        print("Done.")
    finally:
        sys.stdout = old_stdout
        tee.files[1].close()


if __name__ == "__main__":
    main()
