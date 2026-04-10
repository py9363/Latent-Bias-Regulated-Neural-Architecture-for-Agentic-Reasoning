#!/usr/bin/env python3
"""
Bias in Bios **full baseline demo** (same protocol as ``run_all_baselines.py``, with CrowS-Pairs / BBQ).

**Default split:** train/val caps match scale of test ``--train-samples 96 --val-samples 24 --test-samples 32`` (or similar).

**Re-run metrics without retraining:** after a full run, checkpoints live under ``demo/output/checkpoints/``.
Use ``--skip-training`` to load them and only run B0 + test metrics + agentic report (same ``--model``, split caps, and flags as the run that produced the checkpoints).

Runs in order:
  **B0** pretrained probe + task metrics on test  
  **B1** standard fine-tune  
  **B2** adversarial + LoRA (``run_b2``)  
  **B3** INLP (``run_b3``, few iterations)  
  **Main** stability-regularized adversarial (``run_main``)

Checkpoints: ``demo/output/checkpoints/{b1,b2,b3,main}/``  
Log / report: ``demo/output/demo_log.txt``, ``demo/output/demo_full_report.json``, ``demo/output/demo_report.md``  
(``demo_report.md`` = baseline table + agentic B_adv vs Main with **E**-step notation; same style as ``run_all_baselines``.)

Agentic eval runs by default; use ``--skip-table0`` to skip only the biography fine-tune block (memory).

From repo root:
  python demo/run_demo.py

Smaller / faster (tiny split — debiasing metrics may look unstable):
  python demo/run_demo.py --train-samples 96 --val-samples 24 --test-samples 32 --epochs 1 --inlp-k 2 --batch-size 4

Defaults use ``--bios-train-max`` / ``--bios-val-max`` / ``--bios-test-max``; optional ``--train-samples`` /
``--val-samples`` / ``--test-samples`` apply an extra min cap on top.

Skip Main (e.g. very slow on CPU inner loop):
  python demo/run_demo.py --skip-main

Eval only from saved checkpoints (no B1/B2/B3/Main training):
  python demo/run_demo.py --skip-training

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
from adaptation.lora_adaptation import (  # noqa: E402
    extract_hidden_after_lora,
    run_lora_from_baseline_checkpoint,
)
from config import (  # noqa: E402
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
    get_device,
    log_device_banner,
)
from data.adaptation_labels import (  # noqa: E402
    ADAPT_OBJECTIVE_OCCUPATION,
    ADAPT_OBJECTIVE_SUMMARIZE_LM,
)
from data.bias_in_bios import NUM_OCCUPATIONS, load_bias_in_bios  # noqa: E402
from data.loaders import get_qwen_tokenizer, load_bbq, load_crows_pairs  # noqa: E402
from evaluation.capstone_report_md import render_capstone_report_markdown  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    compute_occupation_accuracy_and_gender_gap,
    evaluate_bbq_with_model,
    evaluate_crows_pairs_with_model,
    get_backbone_for_lm,
)
from evaluation.probe import run_probe  # noqa: E402
from run_agentic_baselines import (  # noqa: E402
    DEFAULT_ADAPT_LM_MICRO_BATCH,
    _agentic_eval_for_model,
    _biography_probe_dict,
    _biography_task_finetune_then_Ebio,
    _merge_biography_and_lift,
)
from models.adversarial import QwenAdversarialModel  # noqa: E402
from models.qwen_task import QwenTaskModel  # noqa: E402

# Tiny-split note: very high λ1 can collapse task accuracy on small n; use --full-debias for config.py defaults.
# Demo default λ1=2.0 (stronger adversarial pressure; was 0.45).
DEMO_LAMBDA_BIAS = 2.0
DEMO_LAMBDA_STAB = 0.04
DEMO_MAIN_INNER_STEPS = 4
# Agentic inner loop (demo): stronger than tiny-split training defaults; matches requested eval strength.
DEMO_AGENTIC_ADAPTATION_STEPS = 5
DEMO_AGENTIC_ADAPTATION_LR = 1e-4
# Larger than DEFAULT_BATCH_SIZE (8): more stable training and fewer single-gender batches in A2 dynamic projection.
DEMO_DEFAULT_BATCH_SIZE = max(16, DEFAULT_BATCH_SIZE)


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


def _demo_collect_agentic_eval(
    *,
    train_ds,
    val_ds,
    test_ds,
    tokenizer,
    device: str,
    args: Any,
    ckpt_root: Path,
    lambda_bias: float,
    lambda_stab: float,
    inlp_k: int,
    skip_table0: bool,
    bio_ft_epochs: int,
    main_inner_steps: int,
    adapt_objective: str,
    adapt_lm_max_length: int,
    adapt_lm_micro_batch: int,
    agentic_models: str = "b_adv",
) -> Dict[str, Any]:
    """
    Multi-step agentic metrics + optional TABLE 0. Loads one checkpoint at a time.

    Always evaluates **B_task** (normal fine-tuned B1), **B_adv**, and **Main** (when checkpoint exists and not ``--skip-main``).
    ``agentic_models=all`` adds A2 and B_static_inlp. Returns a dict for ``demo_full_report.json`` / ``demo_report.md``.
    """
    agentic_batch_size = max(1, int(args.agentic_batch_size))
    eval_kw = dict(
        tokenizer=tokenizer,
        dataset=test_ds,
        batch_size=agentic_batch_size,
        max_length=args.max_length,
        device=device,
        seed=args.seed,
        adaptation_steps=DEMO_AGENTIC_ADAPTATION_STEPS,
        adaptation_lr=DEMO_AGENTIC_ADAPTATION_LR,
        recompute_projection_each_step=False,
        lambda_bias_adapt=lambda_bias,
        adaptation_task_only=True,
        adaptation_adapt_lora=True,
        adaptation_grad_clip=1.0,
        adapt_objective=adapt_objective,
        adapt_lm_max_length=adapt_lm_max_length,
        adapt_lm_micro_batch=adapt_lm_micro_batch,
    )
    agentic_results: Dict[str, Dict[str, Any]] = {}
    table0: Dict[str, Dict[str, float]] = {}
    full_agentic = agentic_models.strip().lower() == "all"

    print(
        "\n--- Agentic evaluation (test split, multi-step prompts; B_task + B_adv + Main for report) ---"
        + (" [+ A2, INLP]" if full_agentic else "")
    )
    if agentic_batch_size < args.batch_size:
        print(
            f"  Agentic eval batch_size={agentic_batch_size} (training batch_size={args.batch_size}) to reduce VRAM."
        )

    b1_pt = ckpt_root / "b1_demo" / "pytorch_model.pt"
    m1 = _load_b1_checkpoint(b1_pt, args.model, device, NUM_OCCUPATIONS)
    agentic_results["B_task"] = _agentic_eval_for_model(model=m1, apply_dynamic_reg=False, **eval_kw)
    _merge_biography_and_lift(
        agentic_results["B_task"],
        _biography_probe_dict(m1, test_ds, agentic_batch_size, device, args.seed),
    )
    if not skip_table0:
        _free_cuda()
        table0["B_task"] = _biography_task_finetune_then_Ebio(
            m1,
            train_ds,
            test_ds,
            device,
            agentic_batch_size,
            args.seed,
            ft_epochs=bio_ft_epochs,
            ft_lr=5e-5,
            tokenizer=tokenizer,
            adapt_objective=adapt_objective,
            adapt_lm_max_length=adapt_lm_max_length,
            adapt_lm_micro_batch=adapt_lm_micro_batch,
        )
    if full_agentic:
        agentic_results["A2_runtime_dynamic_proj"] = _agentic_eval_for_model(
            model=m1, apply_dynamic_reg=True, **eval_kw
        )
        _merge_biography_and_lift(
            agentic_results["A2_runtime_dynamic_proj"],
            _biography_probe_dict(m1, test_ds, agentic_batch_size, device, args.seed),
        )
    del m1
    _free_cuda()

    b2_pt = ckpt_root / "b2_demo" / "pytorch_model.pt"
    m2 = load_b2_from_checkpoint(str(b2_pt), args.model, device, NUM_OCCUPATIONS, num_bias_labels=2)
    agentic_results["B_adv"] = _agentic_eval_for_model(model=m2, apply_dynamic_reg=False, **eval_kw)
    _merge_biography_and_lift(
        agentic_results["B_adv"],
        _biography_probe_dict(m2, test_ds, agentic_batch_size, device, args.seed),
    )
    if not skip_table0:
        _free_cuda()
        table0["B_adv"] = _biography_task_finetune_then_Ebio(
            m2,
            train_ds,
            test_ds,
            device,
            agentic_batch_size,
            args.seed,
            ft_epochs=bio_ft_epochs,
            ft_lr=5e-5,
            tokenizer=tokenizer,
            adapt_objective=adapt_objective,
            adapt_lm_max_length=adapt_lm_max_length,
            adapt_lm_micro_batch=adapt_lm_micro_batch,
        )
    del m2
    _free_cuda()

    if full_agentic:
        b3_pt = ckpt_root / "b3_demo" / "pytorch_model.pt"
        m3 = _load_b3_checkpoint(b3_pt, args.model, device, NUM_OCCUPATIONS)
        agentic_results["B_static_inlp"] = _agentic_eval_for_model(model=m3, apply_dynamic_reg=False, **eval_kw)
        _merge_biography_and_lift(
            agentic_results["B_static_inlp"],
            _biography_probe_dict(m3, test_ds, agentic_batch_size, device, args.seed),
        )
        if not skip_table0:
            _free_cuda()
            table0["B_static_inlp"] = _biography_task_finetune_then_Ebio(
                m3,
                train_ds,
                test_ds,
                device,
                agentic_batch_size,
                args.seed,
                ft_epochs=bio_ft_epochs,
                ft_lr=5e-5,
                tokenizer=tokenizer,
                adapt_objective=adapt_objective,
                adapt_lm_max_length=adapt_lm_max_length,
                adapt_lm_micro_batch=adapt_lm_micro_batch,
            )
        del m3
        _free_cuda()

    main_pt = ckpt_root / "main_demo" / "pytorch_model.pt"
    if args.skip_main:
        print("  Agentic Main: skipped (--skip-main).")
    elif not main_pt.is_file():
        print(f"  Agentic Main: skipped — no checkpoint at {main_pt}")
    if not args.skip_main and main_pt.is_file():
        mm = load_main_from_checkpoint(str(main_pt), args.model, device, NUM_OCCUPATIONS, num_bias_labels=2)
        agentic_results["Main"] = _agentic_eval_for_model(model=mm, apply_dynamic_reg=False, **eval_kw)
        _merge_biography_and_lift(
            agentic_results["Main"],
            _biography_probe_dict(mm, test_ds, agentic_batch_size, device, args.seed),
        )
        if not skip_table0:
            _free_cuda()
            table0["Main"] = _biography_task_finetune_then_Ebio(
                mm,
                train_ds,
                test_ds,
                device,
                agentic_batch_size,
                args.seed,
                ft_epochs=bio_ft_epochs,
                ft_lr=5e-5,
                tokenizer=tokenizer,
                adapt_objective=adapt_objective,
                adapt_lm_max_length=adapt_lm_max_length,
                adapt_lm_micro_batch=adapt_lm_micro_batch,
            )
        del mm
        _free_cuda()

    agentic_meta = {
        "timestamp": datetime.now().isoformat(),
        "report_kind": "demo",
        "device": device,
        "model": args.model,
        "seed": args.seed,
        "bios_train": len(train_ds),
        "bios_val": len(val_ds),
        "bios_test": len(test_ds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "agentic_batch_size": agentic_batch_size,
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
        "adaptation_steps": DEMO_AGENTIC_ADAPTATION_STEPS,
        "adaptation_lr": DEMO_AGENTIC_ADAPTATION_LR,
        "adapt_objective": adapt_objective,
        "adapt_lm_max_length": adapt_lm_max_length,
        "adapt_lm_micro_batch": adapt_lm_micro_batch,
        "probe_protocol": (
            "Same sklearn Pipeline(StandardScaler, LogisticRegression) and random_state=seed for R1/R2/R3; "
            "80/20 stratified probe split; scaler fit on probe train only."
        ),
        "baselines_run": list(agentic_results.keys()),
        "agentic_models": agentic_models,
    }

    report = {
        "meta": agentic_meta,
        "results": agentic_results,
        "table0_pure_bio_task_ft": table0,
    }
    return {"report": report}


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


def _attach_crows_bbq_metrics(
    row: Dict[str, Any],
    model,
    tokenizer,
    crows_ds,
    bbq_ds,
    *,
    device: str,
) -> None:
    """Attach CrowS / BBQ metrics to an existing baseline row."""
    if crows_ds is None or bbq_ds is None or len(crows_ds) == 0 or len(bbq_ds) == 0:
        row["crows_pairs_bias_score"] = None
        row["bbq_task_accuracy"] = None
        row["bbq_accuracy_gap"] = None
        return
    try:
        backbone = get_backbone_for_lm(model)
        crows = evaluate_crows_pairs_with_model(
            backbone, tokenizer, crows_ds, device, batch_size=8, max_length=128
        )
        bbq = evaluate_bbq_with_model(
            backbone, tokenizer, bbq_ds, device, batch_size=4, max_length=256
        )
        row["crows_pairs_bias_score"] = round(float(crows["crows_pairs_bias_score"]), 2)
        row["bbq_task_accuracy"] = round(float(bbq["task_accuracy"]), 2)
        row["bbq_accuracy_gap"] = round(float(bbq["bbq_accuracy_gap"]), 2)
    except Exception as e:
        print(f"  Warning: CrowS/BBQ eval failed ({e})")
        row["crows_pairs_bias_score"] = None
        row["bbq_task_accuracy"] = None
        row["bbq_accuracy_gap"] = None


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _demo_ckpt(ckpt_root: Path, sub: str) -> Path:
    return ckpt_root / sub / "pytorch_model.pt"


def _demo_require_ckpt(path: Path, what: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{what} not found: {path}\n"
            "Train once without --skip-training, or fix --checkpoints-dir if you moved checkpoints."
        )


def _demo_metrics_from_saved_checkpoints(
    *,
    ckpt_root: Path,
    args: Any,
    train_ds,
    val_ds,
    test_ds,
    tokenizer,
    crows_ds,
    bbq_ds,
    device: str,
    skip_main: bool,
) -> Dict[str, Dict[str, Any]]:
    """Load B1/B2/B3/(Main) from disk; same metric rows as after training."""
    out: Dict[str, Dict[str, Any]] = {}
    b1_pt = _demo_ckpt(ckpt_root, "b1_demo")
    b2_pt = _demo_ckpt(ckpt_root, "b2_demo")
    b3_pt = _demo_ckpt(ckpt_root, "b3_demo")
    main_pt = _demo_ckpt(ckpt_root, "main_demo")
    _demo_require_ckpt(b1_pt, "B1 checkpoint")
    _demo_require_ckpt(b2_pt, "B2 checkpoint")
    _demo_require_ckpt(b3_pt, "B3 checkpoint")
    if not skip_main:
        _demo_require_ckpt(main_pt, "Main checkpoint")

    print("\n--- B1: Standard fine-tuning (loaded from checkpoint) ---")
    m1 = _load_b1_checkpoint(b1_pt, args.model, device, NUM_OCCUPATIONS)
    out["B1_standard"] = _bios_metrics_row(m1, train_ds, val_ds, test_ds, device, args.batch_size, args.seed)
    _attach_crows_bbq_metrics(out["B1_standard"], m1, tokenizer, crows_ds, bbq_ds, device=device)
    print(f"  occ_acc={out['B1_standard']['occupation_accuracy']}%  R={out['B1_standard']['recoverability_R']}")
    del m1
    _free_cuda()

    print("\n--- B2: Adversarial debiasing (LoRA; loaded from checkpoint) ---")
    m2 = load_b2_from_checkpoint(str(b2_pt), args.model, device, NUM_OCCUPATIONS, num_bias_labels=2)
    out["B2_adversarial"] = _bios_metrics_row(m2, train_ds, val_ds, test_ds, device, args.batch_size, args.seed)
    _attach_crows_bbq_metrics(out["B2_adversarial"], m2, tokenizer, crows_ds, bbq_ds, device=device)
    print(f"  occ_acc={out['B2_adversarial']['occupation_accuracy']}%  R={out['B2_adversarial']['recoverability_R']}")
    del m2
    _free_cuda()

    print(f"\n--- B3: INLP (loaded from checkpoint) ---")
    m3 = _load_b3_checkpoint(b3_pt, args.model, device, NUM_OCCUPATIONS)
    out["B3_INLP"] = _bios_metrics_row(m3, train_ds, val_ds, test_ds, device, args.batch_size, args.seed)
    _attach_crows_bbq_metrics(out["B3_INLP"], m3, tokenizer, crows_ds, bbq_ds, device=device)
    print(f"  occ_acc={out['B3_INLP']['occupation_accuracy']}%  R={out['B3_INLP']['recoverability_R']}")
    del m3
    _free_cuda()

    if not skip_main:
        print("\n--- Main: Stability-regularized adversarial (LoRA; loaded from checkpoint) ---")
        mm = load_main_from_checkpoint(str(main_pt), args.model, device, NUM_OCCUPATIONS, num_bias_labels=2)
        out["Main"] = _bios_metrics_row(mm, train_ds, val_ds, test_ds, device, args.batch_size, args.seed)
        _attach_crows_bbq_metrics(out["Main"], mm, tokenizer, crows_ds, bbq_ds, device=device)
        print(f"  occ_acc={out['Main']['occupation_accuracy']}%  R={out['Main']['recoverability_R']}")
        del mm
        _free_cuda()
    else:
        print("\n--- Main: skipped (--skip-main) ---")

    return out


def _bios_subset_for_lora(dataset, tokenizer, max_length: int, subset_size: int = 1000, pad_token_id: int = 0):
    """Subset of Bias in Bios with LM labels for LoRA adaptation."""
    n = min(subset_size, len(dataset))
    subset = dataset.select(range(n))

    def add_labels(ex):
        ids = ex["input_ids"][:max_length]
        lab = [x if x != pad_token_id else -100 for x in ids]
        return {"labels": lab}

    return subset.map(add_labels, desc="demo labels for LoRA")


def _demo_add_lora_delta_r(
    *,
    bios_results: Dict[str, Dict[str, Any]],
    ckpt_root: Path,
    tokenizer,
    train_ds,
    args: Any,
    device: str,
) -> None:
    """Compute and attach R(theta') and delta_R for demo rows."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    lora_adaptation_dataset = _bios_subset_for_lora(
        train_ds, tokenizer, args.max_length, subset_size=min(1000, len(train_ds)), pad_token_id=pad_id
    )
    baselines_for_lora = [
        ("B0_pretrained", None, "demo_b0"),
        ("B1_standard", str(_demo_ckpt(ckpt_root, "b1_demo")), "demo_b1"),
        ("B2_adversarial", str(_demo_ckpt(ckpt_root, "b2_demo")), "demo_b2"),
        ("B3_INLP", str(_demo_ckpt(ckpt_root, "b3_demo")), "demo_b3"),
    ]
    main_pt = _demo_ckpt(ckpt_root, "main_demo")
    if main_pt.is_file() and "Main" in bios_results:
        baselines_for_lora.append(("Main", str(main_pt), "demo_main"))

    for key, ckpt_path, base_name in baselines_for_lora:
        if key not in bios_results:
            continue
        print(f"\n--- LoRA from {key} (demo) ---")
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
                adapted_model, tokenizer, train_ds, batch_size=args.batch_size, device=device
            )
            probe_adapted = run_probe(
                h_adapted.numpy() if hasattr(h_adapted, "numpy") else h_adapted.cpu().numpy(),
                s_adapted,
                test_size=0.2,
                random_state=args.seed,
            )
            r_theta_prime = round(float(probe_adapted["accuracy"]), 4)
            r_theta = float(bios_results[key]["recoverability_R"])
            delta_r = round(r_theta_prime - r_theta, 4)
            bios_results[key]["R_theta_prime"] = r_theta_prime
            bios_results[key]["delta_R"] = delta_r
            print(f"  {key}: R(θ)={r_theta:.4f} → R(θ')={r_theta_prime:.4f}, ΔR={delta_r:.4f}")
        except Exception as e:
            print(f"  LoRA from {key} failed: {e}")
            bios_results[key]["R_theta_prime"] = None
            bios_results[key]["delta_R"] = None
        if adapted_model is not None:
            del adapted_model
        _free_cuda()


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
        description="Run B0–B3 + Main on Bias in Bios (demo; default caps sized for interpretable debiasing).",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--crows-max",
        type=int,
        default=500,
        help="Max CrowS-Pairs examples for demo eval (default: 500; use larger for more stable estimate)",
    )
    parser.add_argument(
        "--bbq-max",
        type=int,
        default=2000,
        help="Max BBQ examples for demo eval (default: 2000; full split is much larger/slower)",
    )
    parser.add_argument(
        "--bios-train-max",
        type=int,
        default=1000,
        help="Max train bios (predefined train split); tiny train + large test makes B2/B3 misleading",
    )
    parser.add_argument(
        "--bios-val-max",
        type=int,
        default=250,
        help="Max val bios (predefined val split)",
    )
    parser.add_argument(
        "--bios-test-max",
        type=int,
        default=500,
        help="Max test bios (predefined test split) for metrics and agentic eval",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Optional tighter cap on train (min with --bios-train-max); e.g. 96 for quick smoke tests",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=None,
        help="Optional tighter cap on val (min with --bios-val-max)",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Optional tighter cap on test (min with --bios-test-max)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs for B1/B2/B3/Main (default: {DEFAULT_EPOCHS} from config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEMO_DEFAULT_BATCH_SIZE,
        help=f"Train/eval batch size (default {DEMO_DEFAULT_BATCH_SIZE}; use smaller e.g. 4 if GPU OOM)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inlp-k", type=int, default=3, help="INLP iterations (capped vs train size unless --full-debias)")
    parser.add_argument(
        "--full-debias",
        action="store_true",
        help="Use config DEFAULT_LAMBDA_* / DEFAULT_MAIN_INNER_STEPS and raw --inlp-k (harsher; tiny data may collapse task acc).",
    )
    parser.add_argument("--skip-main", action="store_true", help="Skip Main (slow; uses CPU inner twin)")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip B1/B2/B3/Main training; load checkpoints and run B0 + test metrics + agentic eval",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=None,
        help="Checkpoint root (subdirs b1_demo, b2_demo, b3_demo, main_demo). Default: demo/output/checkpoints",
    )
    parser.add_argument(
        "--agentic-models",
        type=str,
        choices=["b_adv", "all"],
        default="b_adv",
        help="Extra agentic models in JSON: b_adv = B_task+B_adv+Main (default); all = also A2, INLP",
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
        default=256,
        help="Max tokens for LM summarize sequences (TABLE 0 + inner loop; lower saves VRAM)",
    )
    parser.add_argument(
        "--adapt-lm-micro-batch",
        type=int,
        default=DEFAULT_ADAPT_LM_MICRO_BATCH,
        help="LM adaptation: forward/backward chunk size (smaller if CUDA OOM during agentic eval)",
    )
    parser.add_argument(
        "--agentic-batch-size",
        type=int,
        default=8,
        help="Batch size for agentic eval + TABLE 0 only (default: 8; lower if OOM with shared GPU).",
    )
    parser.add_argument(
        "--skip-evals",
        action="store_true",
        help="Skip CrowS-Pairs and BBQ by forcing --crows-max 0 --bbq-max 0.",
    )
    args = parser.parse_args()
    if args.skip_evals:
        args.crows_max = 0
        args.bbq_max = 0

    demo_dir = Path(__file__).resolve().parent
    out_dir = demo_dir / "output"
    data_dir = demo_dir / "data"
    ckpt_root = (
        Path(args.checkpoints_dir).expanduser().resolve()
        if args.checkpoints_dir
        else out_dir / "checkpoints"
    )
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
        mode = "EVAL FROM CHECKPOINTS (--skip-training)" if args.skip_training else "FULL BASELINE DEMO (B0, B1, B2, B3, Main)"
        print(f"Bias in Bios — {mode}")
        print(f"UTC: {datetime.now(timezone.utc).isoformat()}")
        print("=" * 72)

        _set_seed(args.seed)
        device = "cpu" if args.cpu else get_device()
        log_device_banner(device)

        tokenizer = get_qwen_tokenizer(args.model)
        train_full, val_full, test_full = load_bias_in_bios(
            tokenizer, max_length=args.max_length, use_predefined_splits=True
        )
        if args.skip_evals:
            crows_ds = None
            bbq_ds = None
        else:
            crows_ds = load_crows_pairs(split="test")
            if args.crows_max is not None and len(crows_ds) > args.crows_max:
                crows_ds = crows_ds.select(range(args.crows_max))
            bbq_ds = load_bbq(config=None, split="test")
            if args.bbq_max is not None and len(bbq_ds) > args.bbq_max:
                bbq_ds = bbq_ds.select(range(args.bbq_max))

        n_tr = min(args.bios_train_max, len(train_full))
        if args.train_samples is not None:
            n_tr = min(n_tr, args.train_samples)
        n_va = min(args.bios_val_max, len(val_full))
        if args.val_samples is not None:
            n_va = min(n_va, args.val_samples)
        n_te = min(args.bios_test_max, len(test_full))
        if args.test_samples is not None:
            n_te = min(n_te, args.test_samples)
        train_ds = train_full.select(range(n_tr))
        val_ds = val_full.select(range(n_va))
        test_ds = test_full.select(range(n_te))

        print(f"\nDemo split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
        crows_n = 0 if crows_ds is None else len(crows_ds)
        bbq_n = 0 if bbq_ds is None else len(bbq_ds)
        print(f"Eval sets: CrowS={crows_n} BBQ={bbq_n}")
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
                "agentic_batch_size": args.agentic_batch_size,
                "lora_post_adaptation": True,
                "seed": args.seed,
                "inlp_k_requested": args.inlp_k,
                "inlp_k_used": inlp_k,
                "lambda_bias": lambda_bias,
                "lambda_stab": lambda_stab,
                "main_inner_steps": main_inner_steps,
                "full_debias": args.full_debias,
                "skip_main": args.skip_main,
                "skip_table0": args.skip_table0,
                "agentic_bio_ft_epochs": args.agentic_bio_ft_epochs,
                "bias_loss_balance": not args.no_bias_loss_balance,
                "adapt_objective": args.adapt_objective,
                "adapt_lm_max_length": args.adapt_lm_max_length,
                "adapt_lm_micro_batch": args.adapt_lm_micro_batch,
                "bios_train_max": args.bios_train_max,
                "bios_val_max": args.bios_val_max,
                "bios_test_max": args.bios_test_max,
                "crows_examples": crows_n,
                "bbq_examples": bbq_n,
                "train_samples_override": args.train_samples,
                "val_samples_override": args.val_samples,
                "test_samples_override": args.test_samples,
                "agentic_adaptation_steps": DEMO_AGENTIC_ADAPTATION_STEPS,
                "agentic_adaptation_lr": DEMO_AGENTIC_ADAPTATION_LR,
                "agentic_models": args.agentic_models,
                "skip_training": args.skip_training,
                "checkpoints_dir": str(ckpt_root),
            }
        )

        # ---- B0: pretrained (no training) ----
        print("\n--- B0: Pretrained Qwen (no fine-tuning) ---")
        model_b0 = QwenTaskModel(model_name=args.model, num_labels=NUM_OCCUPATIONS).to(device)
        model_b0.eval()
        bios_results["B0_pretrained"] = _bios_metrics_row(
            model_b0, train_ds, val_ds, test_ds, device, args.batch_size, args.seed
        )
        _attach_crows_bbq_metrics(bios_results["B0_pretrained"], model_b0, tokenizer, crows_ds, bbq_ds, device=device)
        print(
            f"  occ_acc={bios_results['B0_pretrained']['occupation_accuracy']}%  "
            f"R={bios_results['B0_pretrained']['recoverability_R']}"
        )
        del model_b0
        _free_cuda()

        if args.skip_training:
            print(f"\n--- --skip-training: loading checkpoints from {ckpt_root} ---")
            trained = _demo_metrics_from_saved_checkpoints(
                ckpt_root=ckpt_root,
                args=args,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                tokenizer=tokenizer,
                crows_ds=crows_ds,
                bbq_ds=bbq_ds,
                device=device,
                skip_main=args.skip_main,
            )
            bios_results.update(trained)
        else:
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
            _attach_crows_bbq_metrics(bios_results["B1_standard"], model_b1, tokenizer, crows_ds, bbq_ds, device=device)
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
            _attach_crows_bbq_metrics(bios_results["B2_adversarial"], model_b2, tokenizer, crows_ds, bbq_ds, device=device)
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
            _attach_crows_bbq_metrics(bios_results["B3_INLP"], model_b3, tokenizer, crows_ds, bbq_ds, device=device)
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
                _attach_crows_bbq_metrics(bios_results["Main"], model_m, tokenizer, crows_ds, bbq_ds, device=device)
                print(
                    f"  occ_acc={bios_results['Main']['occupation_accuracy']}%  "
                    f"R={bios_results['Main']['recoverability_R']}"
                )
                del model_m
                _free_cuda()
            else:
                print("\n--- Main: skipped (--skip-main) ---")

        # ---- LoRA from each baseline: R(theta'), delta_R ----
        _demo_add_lora_delta_r(
            bios_results=bios_results,
            ckpt_root=ckpt_root,
            tokenizer=tokenizer,
            train_ds=train_ds,
            args=args,
            device=device,
        )

        bio_ft_epochs = (
            args.agentic_bio_ft_epochs
            if args.agentic_bio_ft_epochs is not None
            else (2 if len(train_ds) < 200 else 3)
        )
        agentic_bundle: Optional[Dict[str, Any]] = _demo_collect_agentic_eval(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            tokenizer=tokenizer,
            device=device,
            args=args,
            ckpt_root=ckpt_root,
            lambda_bias=lambda_bias,
            lambda_stab=lambda_stab,
            inlp_k=inlp_k,
            skip_table0=args.skip_table0,
            bio_ft_epochs=bio_ft_epochs,
            main_inner_steps=main_inner_steps,
            adapt_objective=args.adapt_objective,
            adapt_lm_max_length=args.adapt_lm_max_length,
            adapt_lm_micro_batch=args.adapt_lm_micro_batch,
            agentic_models=args.agentic_models,
        )

        _print_results_table(bios_results)

        ts_utc = datetime.now(timezone.utc).isoformat()
        meta["timestamp"] = ts_utc
        baseline_meta_for_md = {
            "timestamp": ts_utc,
            "model": meta["model"],
            "device": meta["device"],
            "seed": meta["seed"],
            "bios_train": meta.get("train_n"),
            "bios_val": meta.get("val_n"),
            "bios_test": meta.get("test_n"),
            "crows_examples": meta.get("crows_examples"),
            "bbq_examples": meta.get("bbq_examples"),
        }
        ar = None if agentic_bundle is None else agentic_bundle["report"]
        demo_md_path = out_dir / "demo_report.md"
        md_demo = render_capstone_report_markdown(
            baseline_meta_for_md,
            bios_results,
            agentic_meta=ar["meta"] if ar else None,
            agentic_results=ar["results"] if ar else None,
            table0=ar["table0_pure_bio_task_ft"] if ar else None,
            skip_table0=args.skip_table0,
            title="Demo report (Bias in Bios; no CrowS/BBQ)",
        )
        with open(demo_md_path, "w", encoding="utf-8") as f:
            f.write(md_demo)

        out_payload = {
            "timestamp_utc": ts_utc,
            "meta": meta,
            "results": bios_results,
            "agentic_report": ar,
            "report_markdown": str(demo_md_path),
            "checkpoints_dir": str(ckpt_root),
            "log_path": str(log_path),
            "sample_batch_path": str(data_dir / "sample_batch.json"),
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2)
        print(f"\nWrote JSON report → {report_path}")
        print(f"Report (Markdown) → {demo_md_path}")
        print(f"Full log → {log_path}")
        print("Done.")
    finally:
        sys.stdout = old_stdout
        tee.files[1].close()


if __name__ == "__main__":
    main()
