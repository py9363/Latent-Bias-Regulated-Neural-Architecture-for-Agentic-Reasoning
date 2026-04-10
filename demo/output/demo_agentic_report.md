# Agentic Baseline Report (Demo — B_adv only)

## Three claims (paper spine)

| Claim | What to show | Tables |
|-------|----------------|--------|
| **1 — Debiasing works (initially)** | Lower excess recoverability on **biography** inputs: **E_bio** (B_adv, INLP) < B_task | **TABLE 1** |
| **2 — Bias returns** | **TABLE 0:** after **task-shift adaptation** on bios (default: **LM summarize**), **E_after > E_before**. **TABLE 2–3:** agentic lift + step drift | **TABLE 0** (isolated), **TABLE 2–3** |
| **3 — Main stabilizes** | After you train **Main**: **E1 ≈ E_bio**, **E3 ≈ E1**, small **ΔE**; task accuracy not collapsed | **TABLE 4** + **TABLE 5** |

### TABLE 0 — Pure fine-tuning effect (no prompt change) ⭐

Same tokenized biographies: extra **task loss** on the **train** split (bias head frozen for B_adv), then re-probe **test** bios for **gender** (unchanged linear probe on pooled states). **Positive ΔE** on B_adv → representation shift revives recoverable bias under adaptation. *Default **task shift**: causal LM on **"Summarize this biography."** + biography + a short pseudo-summary target; classification heads stay frozen; backbone (e.g. LoRA) moves — gender probe on pooled h is unchanged.*

| Model | E_before | E_after | ΔE (after − before) |
|-------|----------|---------|---------------------|
| B_adv | 0.8884 | 0.6652 | -0.2232 |

**Generated:** 2026-04-05T20:34:47.519627
**Model:** Qwen/Qwen2.5-0.5B | **Device:** cuda:9 | **Seed:** 42
**Data:** Bios train=1000 val=250 test=500
**λ1 (CLI):** 0.45 | **λ1 used to train B_adv / Main:** 0.45 | **λ2 (Main):** 0.04 | **INLP k used:** 3

### TABLE 1 — Biography (Claim 1: suppression on training distribution)

| Model | R_bio | E_bio (↓ better) | Notes |
|-------|-------|------------------|-------|
| B_adv | 0.95 | 0.8884 | Adversarial suppression |

*Target:* **B_adv** (and/or INLP) **E_bio** clearly **< B_task**; sweet spot often **0.3 < E_bio < 0.8** (tune `--lambda-bias`, `--inlp-iterations`, or `--weak-debias`).

### TABLE 2 — Bias return after agentic step 1 (Claim 2a)

| Model | E_bio | E1 | E1 − E_bio (lift; + = return / shift) |
|-------|-------|-----|----------------------------------------|
| B_adv | 0.8884 | 0.7768 | -0.1116 |

*Target:* **B_adv** with **E1 > E_bio** (positive lift) → bias suppressed on bios but **re-emerges** under agentic prompting + inner adaptation.

### TABLE 3 — Drift across reasoning steps (Claim 2b)

| Model | E1 | E3 | ΔE = E3 − E1 |
|-------|-----|-----|--------------|
| B_adv | 0.7768 | 0.7991 | 0.0223 |

*Target:* **B_adv**: **ΔE > 0** (bias accumulates from step 1 → step 3).

### TABLE 4 — Main vs B_adv (Claim 3)

| Model | E_bio | E1 | E3 | ΔE (E3−E1) |
|-------|-------|-----|-----|------------|
| B_adv | 0.8884 | 0.7768 | 0.7991 | 0.0223 |

### TABLE 5 — Task utility (final agentic step occupation accuracy %)

| Model | Final step occ. acc % |
|-------|------------------------|
| B_adv | 46.0 |

---

## Full metric dump (all baselines)

| Baseline | Final Occ Acc % | R1 | R2 | R3 | ΔR | E1 | E2 | E3 | ΔE |
|----------|------------------|----|----|----|----|----|----|----|-----|
| B_adv | 46.0 | 0.9 | 0.91 | 0.91 | 0.01 | 0.7768 | 0.7991 | 0.7991 | 0.0223 |

R1/R2/R3: raw probe accuracy; E1/E2/E3: excess recoverability; ΔE = E3−E1. See `meta.probe_protocol`.
Inner adaptation: B_task → L_task on task head. B_adv/Main (LoRA): L_task on LoRA+task head (default); set CONFIG `adaptation_task_only=False` for L_task+λ1 L_bias on bias head too.

## Supporting: full biography + lift (all rows)
| Baseline | R_bio | E_bio | ROC-AUC | E1−E_bio |
|----------|-------|-------|---------|----------|
| B_adv | 0.95 | 0.8884 | 0.9863 | -0.1116 |