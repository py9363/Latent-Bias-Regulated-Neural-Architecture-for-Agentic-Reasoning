"""Compact capstone report: baseline metrics table + agentic (B_adv vs Main) with E-step notation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Keys as stored in run_all_baselines / demo JSON
BASELINE_ORDER: List[str] = [
    "B0_pretrained",
    "B1_standard",
    "B2_adversarial",
    "B3_INLP",
    "Main",
]

BASELINE_DISPLAY: Dict[str, str] = {
    "B0_pretrained": "B0 pretrained",
    "B1_standard": "B1 standard",
    "B2_adversarial": "B2 adversarial",
    "B3_INLP": "B3 INLP",
    "Main": "Main",
}


def _cell(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        if nd == 2:
            return f"{float(v):.2f}"
        if nd == 4:
            return f"{float(v):.4f}"
        return str(v)
    return str(v)


def render_baseline_markdown_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Markdown table: Acc / Gap / R(θ) / R(θ′) / ΔR / CrowS / BBQ Acc."""
    lines = [
        "## Baseline results",
        "",
        "| Baseline | Acc (%) | Gap (%) | R(θ) | R(θ′) | ΔR | CrowS (%) | BBQ Acc |",
        "|----------|---------|---------|------|-------|-----|-----------|---------|",
    ]
    for key in BASELINE_ORDER:
        if key not in results:
            continue
        r = results[key]
        lines.append(
            "| "
            + " | ".join(
                [
                    BASELINE_DISPLAY.get(key, key),
                    _cell(r.get("occupation_accuracy"), nd=2),
                    _cell(r.get("gender_gap"), nd=4),
                    _cell(r.get("recoverability_R"), nd=4),
                    _cell(r.get("R_theta_prime"), nd=4),
                    _cell(r.get("delta_R"), nd=4),
                    _cell(r.get("crows_pairs_bias_score"), nd=2),
                    _cell(r.get("bbq_task_accuracy"), nd=2),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "*R(θ)*: gender recoverability (linear probe on pooled representations). "
            "*R(θ′)* and *ΔR*: after LoRA task adaptation on Bios. "
            "CrowS / BBQ accuracy omitted unless those metrics are present.",
        ]
    )
    return "\n".join(lines)


def _agentic_row_md(label: str, r: Dict[str, Any]) -> str:
    return (
        "| "
        + " | ".join(
            [
                label,
                _cell(r.get("biography_probe_E"), nd=4),
                _cell(r.get("step1_excess_recoverability_E1"), nd=4),
                _cell(r.get("step2_excess_recoverability_E2"), nd=4),
                _cell(r.get("step3_excess_recoverability_E3"), nd=4),
                _cell(r.get("trajectory_delta_excess_R"), nd=4),
                _cell(r.get("final_step_occupation_accuracy"), nd=2),
            ]
        )
        + " |"
    )


def render_agentic_b_adv_vs_main_markdown(
    meta: Dict[str, Any],
    results: Dict[str, Dict[str, Any]],
    table0: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    skip_table0: bool = False,
) -> str:
    """
    Agentic comparison: **E** = excess recoverability at biography probe (E_bio) or after inner
    adaptation at multi-step agentic prompts (E₁, E₂, E₃). ΔE = E₃ − E₁ (drift across steps).
    """
    lines = [
        "## Agentic evaluation (adversarial vs this work)",
        "",
        "Multi-step prompts on the Bios **test** split; each step runs **inner** gradient steps on the batch, "
        "then a linear gender probe measures recoverability. "
        "**E_bio**: excess recoverability on plain biographies (no agentic prefix). "
        "**E₁, E₂, E₃**: excess recoverability after inner adaptation at agentic reasoning steps 1–3. "
        "**ΔE** = E₃ − E₁. Final column: occupation accuracy at the last step (%).",
        "",
    ]
    inner_steps = meta.get("adaptation_steps", "—")
    inner_lr = meta.get("adaptation_lr", "—")
    lines.append(
        f"*Inner adaptation:* {inner_steps} steps, lr={inner_lr} (see `meta` in JSON for full flags)."
    )
    lines.append("")

    lines.extend(
        [
            "| Model | E_bio | E₁ | E₂ | E₃ | ΔE | Final occ acc (%) |",
            "|-------|-------|----|----|----|----|-------------------|",
        ]
    )
    if "B_task" in results:
        lines.append(_agentic_row_md("Agentic standard (B1)", results["B_task"]))
    else:
        lines.append("| Agentic standard (B1) | — | — | — | — | — | — |")
    if "B_adv" in results:
        lines.append(_agentic_row_md("Agentic adversarial (B2)", results["B_adv"]))
    else:
        lines.append("| Agentic adversarial (B2) | — | — | — | — | — | — |")

    if "Main" in results:
        lines.append(_agentic_row_md("Agentic this work (Main)", results["Main"]))
    else:
        lines.append("| Agentic this work (Main) | — | — | — | — | — | — |")

    if not skip_table0 and table0:
        lines.extend(
            [
                "",
                "### Biography fine-tuning shift (E on train bios, probe on test)",
                "",
                "| Model | E before FT | E after FT | ΔE_FT | PPL before FT | PPL after FT | ΔPPL_FT |",
                "|-------|-------------|------------|-------|---------------|--------------|---------|",
            ]
        )
        for model_key, display in (
            ("B_task", "Agentic standard (B1)"),
            ("B_adv", "Agentic adversarial (B2)"),
            ("Main", "Agentic this work (Main)"),
        ):
            if model_key not in table0:
                continue
            t = table0[model_key]
            lines.append(
                "| "
                + " | ".join(
                    [
                        display,
                        _cell(t.get("E_bio_before_ft"), nd=4),
                        _cell(t.get("E_bio_after_bio_ft"), nd=4),
                        _cell(t.get("delta_E_bio_pure_ft"), nd=4),
                        _cell(t.get("lm_ppl_before_ft"), nd=4),
                        _cell(t.get("lm_ppl_after_ft"), nd=4),
                        _cell(t.get("delta_lm_ppl_ft"), nd=4),
                    ]
                )
                + " |"
            )

    lines.append("")
    return "\n".join(lines)


def render_capstone_report_markdown(
    baseline_meta: Dict[str, Any],
    baseline_results: Dict[str, Dict[str, Any]],
    *,
    agentic_meta: Optional[Dict[str, Any]] = None,
    agentic_results: Optional[Dict[str, Dict[str, Any]]] = None,
    table0: Optional[Dict[str, Dict[str, Any]]] = None,
    skip_table0: bool = False,
    title: str = "Experiment report",
) -> str:
    parts = [
        f"# {title}",
        "",
        f"**Generated:** {baseline_meta.get('timestamp', '—')}",
        f"**Model:** {baseline_meta.get('model', '—')} | **Device:** {baseline_meta.get('device', '—')} | **Seed:** {baseline_meta.get('seed', '—')}",
    ]
    bt = baseline_meta.get("bios_train")
    bv = baseline_meta.get("bios_val")
    bte = baseline_meta.get("bios_test")
    if bt is not None:
        parts.append(f"**Bias in Bios:** train={bt} val={bv} test={bte}")
    ce = baseline_meta.get("crows_examples")
    be = baseline_meta.get("bbq_examples")
    if ce is not None:
        parts.append(f"**CrowS-Pairs:** {ce} | **BBQ:** {be}")
    parts.append("")
    parts.append(render_baseline_markdown_table(baseline_results))

    if agentic_results is not None and agentic_meta is not None:
        parts.append("")
        parts.append(render_agentic_b_adv_vs_main_markdown(agentic_meta, agentic_results, table0, skip_table0=skip_table0))

    return "\n".join(parts)
