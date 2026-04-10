# Demo report (Bias in Bios; no CrowS/BBQ)

**Generated:** 2026-04-07T11:40:48.943534+00:00
**Model:** Qwen/Qwen2.5-0.5B | **Device:** cuda:9 | **Seed:** 42
**Bias in Bios:** train=500 val=150 test=250
**CrowS-Pairs:** 0 | **BBQ:** 0

## Baseline results

| Baseline | Acc (%) | Gap (%) | R(θ) | R(θ′) | ΔR | CrowS (%) | BBQ Acc |
|----------|---------|---------|------|-------|-----|-----------|---------|
| B0 pretrained | 7.20 | 2.1745 | 0.9692 | 0.9800 | 0.0108 | — | — |
| B1 standard | 70.00 | 5.1467 | 0.9692 | 0.9800 | 0.0108 | — | — |
| B2 adversarial | 44.80 | 1.6598 | 0.8846 | 0.9900 | 0.1054 | — | — |
| B3 INLP | 33.60 | 4.8636 | 0.9154 | 0.9800 | 0.0646 | — | — |
| Main | 47.60 | 1.2609 | 0.9308 | 0.9800 | 0.0492 | — | — |

*R(θ)*: gender recoverability (linear probe on pooled representations). *R(θ′)* and *ΔR*: after LoRA task adaptation on Bios. CrowS / BBQ accuracy omitted unless those metrics are present.

## Agentic evaluation (adversarial vs this work)

Multi-step prompts on the Bios **test** split; each step runs **inner** gradient steps on the batch, then a linear gender probe measures recoverability. **E_bio**: excess recoverability on plain biographies (no agentic prefix). **E₁, E₂, E₃**: excess recoverability after inner adaptation at agentic reasoning steps 1–3. **ΔE** = E₃ − E₁. Final column: occupation accuracy at the last step (%).

*Inner adaptation:* 5 steps, lr=0.0001 (see `meta` in JSON for full flags).

| Model | E_bio | E₁ | E₂ | E₃ | ΔE | Final occ acc (%) |
|-------|-------|----|----|----|----|-------------------|
| Agentic standard (B1) | 0.6983 | 0.7845 | 0.7845 | 0.7414 | -0.0431 | 26.40 |
| Agentic adversarial (B2) | 0.8276 | 0.3534 | 0.2672 | 0.2672 | -0.0862 | 37.60 |
| Agentic this work (Main) | 0.7845 | 0.4828 | 0.3534 | 0.3103 | -0.1724 | 41.60 |

### Biography fine-tuning shift (E on train bios, probe on test)

| Model | E before FT | E after FT | ΔE_FT | PPL before FT | PPL after FT | ΔPPL_FT |
|-------|-------------|------------|-------|---------------|--------------|---------|
| Agentic standard (B1) | 0.6983 | 0.8276 | 0.1293 | 1.6390 | 1.0071 | -0.6319 |
| Agentic adversarial (B2) | 0.8276 | 0.7845 | -0.0431 | 2.0526 | 1.0031 | -1.0495 |
| Agentic this work (Main) | 0.7845 | 0.6983 | -0.0862 | 2.2956 | 1.0655 | -1.2300 |
