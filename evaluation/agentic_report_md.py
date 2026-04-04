"""Shared Markdown body for agentic baseline reports (full runs and demo)."""
from __future__ import annotations

from typing import Dict, List, Optional


def render_agentic_report_markdown(
    meta: Dict,
    results: Dict[str, Dict],
    table0_pure_bio_task_ft: Dict[str, Dict],
    *,
    skip_biography_probe: bool = False,
    paper_core: Optional[List[str]] = None,
    paper_support: Optional[List[str]] = None,
    report_title: str = "Agentic Baseline Report",
) -> str:
    paper_core = paper_core or ["B_task", "B_adv", "B_static_inlp"]
    paper_support = paper_support or ["A2_runtime_dynamic_proj"]

    lines = [
        f"# {report_title}",
        "",
        "## Three claims (paper spine)",
        "",
        "| Claim | What to show | Tables |",
        "|-------|----------------|--------|",
        "| **1 — Debiasing works (initially)** | Lower excess recoverability on **biography** inputs: **E_bio** (B_adv, INLP) < B_task | **TABLE 1** |",
        "| **2 — Bias returns** | **TABLE 0:** after **task-shift adaptation** on bios (default: **LM summarize**), **E_after > E_before**. **TABLE 2–3:** agentic lift + step drift | **TABLE 0** (isolated), **TABLE 2–3** |",
        "| **3 — Main stabilizes** | After you train **Main**: **E1 ≈ E_bio**, **E3 ≈ E1**, small **ΔE**; task accuracy not collapsed | **TABLE 4** + **TABLE 5** |",
        "",
        "### TABLE 0 — Pure fine-tuning effect (no prompt change) ⭐",
        "",
    ]
    _t0_body = (
        "Same tokenized biographies: extra **task loss** on the **train** split (bias head frozen for B_adv), "
        "then re-probe **test** bios for **gender** (unchanged linear probe on pooled states). "
        "**Positive ΔE** on B_adv → representation shift revives recoverable bias under adaptation. "
    )
    if meta.get("adapt_objective") == "occupation":
        _t0_body += (
            "*This run uses the **same** objective as debiasing (28-way occupation), so drift is often small.*"
        )
    else:
        _t0_body += (
            "*Default **task shift**: causal LM on **\"Summarize this biography.\"** + biography + a short pseudo-summary target; "
            "classification heads stay frozen; backbone (e.g. LoRA) moves — gender probe on pooled h is unchanged.*"
        )
    lines.extend(
        [
            _t0_body,
            "",
            "| Model | E_before | E_after | ΔE (after − before) |",
            "|-------|----------|---------|---------------------|",
        ]
    )
    for name in paper_core:
        if name in table0_pure_bio_task_ft:
            t0 = table0_pure_bio_task_ft[name]
            lines.append(
                f"| {name} | {t0['E_bio_before_ft']} | {t0['E_bio_after_bio_ft']} | {t0['delta_E_bio_pure_ft']} |"
            )
    if not table0_pure_bio_task_ft:
        lines.append("| *skipped* | — | — | — |")
    lines.extend([
        "",
        f"**Generated:** {meta.get('timestamp', '—')}",
        f"**Model:** {meta.get('model', '—')} | **Device:** {meta.get('device', '—')} | **Seed:** {meta.get('seed', '—')}",
        f"**Data:** Bios train={meta.get('bios_train', '—')} val={meta.get('bios_val', '—')} test={meta.get('bios_test', '—')}",
        f"**λ1 (CLI):** {meta.get('lambda_bias', '—')} | **λ1 used to train B_adv / Main:** {meta.get('lambda_bias_train_used', meta.get('lambda_bias', '—'))} | "
        f"**λ2 (Main):** {meta.get('lambda_stab', '—')} | **INLP k used:** {meta.get('inlp_iterations_used', '—')}",
        "",
        "### TABLE 1 — Biography (Claim 1: suppression on training distribution)",
        "",
        "| Model | R_bio | E_bio (↓ better) | Notes |",
        "|-------|-------|------------------|-------|",
    ])
    if skip_biography_probe:
        lines.append("| *skipped* | — | — | Re-run without skipping biography probe |")
    else:
        for name in paper_core:
            if name not in results:
                continue
            r = results[name]
            note = "High bias (reference)" if name == "B_task" else ("Adversarial suppression" if name == "B_adv" else "Static INLP")
            lines.append(
                f"| {name} | {r['biography_probe_R']} | {r['biography_probe_E']} | {note} |"
            )
    lines.extend([
        "",
        "*Target:* **B_adv** (and/or INLP) **E_bio** clearly **< B_task**; sweet spot often **0.3 < E_bio < 0.8** (tune `--lambda-bias`, `--inlp-iterations`, or `--weak-debias`).",
        "",
        "### TABLE 2 — Bias return after agentic step 1 (Claim 2a)",
        "",
        "| Model | E_bio | E1 | E1 − E_bio (lift; + = return / shift) |",
        "|-------|-------|-----|----------------------------------------|",
    ])
    if skip_biography_probe:
        lines.append("| *skipped* | — | — | — |")
    else:
        for name in paper_core:
            if name not in results:
                continue
            r = results[name]
            lines.append(
                f"| {name} | {r['biography_probe_E']} | {r['step1_excess_recoverability_E1']} | {r['agentic_E1_minus_biography_E']} |"
            )
    lines.extend([
        "",
        "*Target:* **B_adv** with **E1 > E_bio** (positive lift) → bias suppressed on bios but **re-emerges** under agentic prompting + inner adaptation.",
        "",
        "### TABLE 3 — Drift across reasoning steps (Claim 2b)",
        "",
        "| Model | E1 | E3 | ΔE = E3 − E1 |",
        "|-------|-----|-----|--------------|",
    ])
    for name in paper_core:
        if name not in results:
            continue
        r = results[name]
        lines.append(
            f"| {name} | {r['step1_excess_recoverability_E1']} | {r['step3_excess_recoverability_E3']} | {r['trajectory_delta_excess_R']} |"
        )
    lines.extend([
        "",
        "*Target:* **B_adv**: **ΔE > 0** (bias accumulates from step 1 → step 3).",
        "",
        "### TABLE 4 — Main vs B_adv (Claim 3)",
        "",
        "| Model | E_bio | E1 | E3 | ΔE (E3−E1) |",
        "|-------|-------|-----|-----|------------|",
    ])
    if "B_adv" in results:
        r = results["B_adv"]
        if not skip_biography_probe:
            lines.append(
                f"| B_adv | {r['biography_probe_E']} | {r['step1_excess_recoverability_E1']} | "
                f"{r['step3_excess_recoverability_E3']} | {r['trajectory_delta_excess_R']} |"
            )
        else:
            lines.append(
                f"| B_adv | — | {r['step1_excess_recoverability_E1']} | "
                f"{r['step3_excess_recoverability_E3']} | {r['trajectory_delta_excess_R']} |"
            )
    if "Main" in results:
        rm = results["Main"]
        if not skip_biography_probe:
            lines.append(
                f"| Main | {rm['biography_probe_E']} | {rm['step1_excess_recoverability_E1']} | "
                f"{rm['step3_excess_recoverability_E3']} | {rm['trajectory_delta_excess_R']} |"
            )
        else:
            lines.append(
                f"| Main | — | {rm['step1_excess_recoverability_E1']} | "
                f"{rm['step3_excess_recoverability_E3']} | {rm['trajectory_delta_excess_R']} |"
            )
    else:
        lines.append(
            "| **Main** | *train (`run_main`) or add `checkpoints/main/pytorch_model.pt`* | — | — | — |"
        )
    lines.extend([
        "",
        "### TABLE 5 — Task utility (final agentic step occupation accuracy %)",
        "",
        "| Model | Final step occ. acc % |",
        "|-------|------------------------|",
    ])
    for name in paper_core + paper_support:
        if name not in results:
            continue
        r = results[name]
        lines.append(f"| {name} | {r['final_step_occupation_accuracy']} |")
    if "Main" in results:
        lines.append(f"| Main | {results['Main']['final_step_occupation_accuracy']} |")
    else:
        lines.append("| **Main** | *not evaluated (no checkpoint / skipped train)* |")
    lines.extend([
        "",
        "---",
        "",
        "## Full metric dump (all baselines)",
        "",
        "| Baseline | Final Occ Acc % | R1 | R2 | R3 | ΔR | E1 | E2 | E3 | ΔE |",
        "|----------|------------------|----|----|----|----|----|----|----|-----|",
    ])
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['final_step_occupation_accuracy']} | {r['step1_recoverability_R1']} | "
            f"{r['step2_recoverability_R2']} | {r['step3_recoverability_R3']} | {r['trajectory_delta_R']} | "
            f"{r['step1_excess_recoverability_E1']} | {r['step2_excess_recoverability_E2']} | {r['step3_excess_recoverability_E3']} | "
            f"{r['trajectory_delta_excess_R']} |"
        )
    lines.append("")
    lines.append("R1/R2/R3: raw probe accuracy; E1/E2/E3: excess recoverability; ΔE = E3−E1. See `meta.probe_protocol`.")
    lines.append(
        "Inner adaptation: B_task → L_task on task head. B_adv/Main (LoRA): L_task on LoRA+task head (default); "
        "set CONFIG `adaptation_task_only=False` for L_task+λ1 L_bias on bias head too."
    )
    if not skip_biography_probe:
        lines.append("")
        lines.append("## Supporting: full biography + lift (all rows)")
        lines.append("| Baseline | R_bio | E_bio | ROC-AUC | E1−E_bio |")
        lines.append("|----------|-------|-------|---------|----------|")
        for name, r in results.items():
            roc = r.get("biography_probe_roc_auc", "—")
            lines.append(
                f"| {name} | {r['biography_probe_R']} | {r['biography_probe_E']} | {roc} | "
                f"{r['agentic_E1_minus_biography_E']} |"
            )
    return "\n".join(lines)
