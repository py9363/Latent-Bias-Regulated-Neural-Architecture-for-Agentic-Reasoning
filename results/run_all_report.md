# Run-all report (Bias in Bios + CrowS + BBQ + agentic)

**Generated:** 2026-04-09T16:17:58.737263
**Model:** Qwen/Qwen2.5-0.5B | **Device:** cuda:9 | **Seed:** 42
**Bias in Bios:** train=5000 val=1250 test=2500
**CrowS-Pairs:** 500 | **BBQ:** 2000

## Baseline results


| Baseline       | Acc (%) | Gap (%) | R(θ)   | R(θ′)  | ΔR     | CrowS (%) | BBQ Acc |
| -------------- | ------- | ------- | ------ | ------ | ------ | --------- | ------- |
| B0 pretrained  | 6.24    | 2.6213  | 0.9830 | 0.9840 | 0.0010 | 52.40     | 38.50   |
| B1 standard    | 79.08   | 1.1435  | 0.9648 | 0.9780 | 0.0132 | 54.60     | 31.35   |
| B2 adversarial | 72.92   | 0.6039  | 0.8790 | 0.9750 | 0.0960 | 54.20     | 46.25   |
| B3 INLP        | 54.64   | 4.6767  | 0.9280 | 0.9790 | 0.0510 | 52.40     | 38.50   |
| Main           | 74.16   | 0.1462  | 0.8550 | 0.9810 | 0.1260 | 49.80     | 46.90   |


*R(θ)*: gender recoverability (linear probe on pooled representations). *R(θ′)* and *ΔR*: after LoRA task adaptation on Bios. CrowS / BBQ accuracy omitted unless those metrics are present.

## Agentic evaluation (adversarial vs this work)

Multi-step prompts on the Bios **test** split; each step runs **inner** gradient steps on the batch, then a linear gender probe measures recoverability. **E_bio**: excess recoverability on plain biographies (no agentic prefix). **E₁, E₂, E₃**: excess recoverability after inner adaptation at agentic reasoning steps 1–3. **ΔE** = E₃ − E₁. Final column: occupation accuracy at the last step (%).

*Inner adaptation:* 5 steps, lr=0.0001 (see `meta` in JSON for full flags).


| Model                    | E_bio  | E₁     | E₂     | E₃     | ΔE      | Final occ acc (%) |
| ------------------------ | ------ | ------ | ------ | ------ | ------- | ----------------- |
| Agentic standard (B1)    | 0.9387 | 0.8687 | 0.8730 | 0.8643 | -0.0044 | 26.00             |
| Agentic adversarial (B2) | 0.7504 | 0.4658 | 0.5053 | 0.4746 | 0.0088  | 54.52             |
| Agentic this work (Main) | 0.7067 | 0.5096 | 0.4308 | 0.4877 | -0.0219 | 64.04             |


### Biography fine-tuning shift (E on train bios, probe on test)


| Model                    | E before FT | E after FT | ΔE_FT  | PPL before FT | PPL after FT | ΔPPL_FT  |
| ------------------------ | ----------- | ---------- | ------ | ------------- | ------------ | -------- |
| Agentic standard (B1)    | 0.9387      | 0.9518     | 0.0131 | 1.6193        | 1.0031       | -0.6162  |
| Agentic adversarial (B2) | 0.7504      | 0.9037     | 0.1533 | 11.2611       | 1.0018       | -10.2594 |
| Agentic this work (Main) | 0.7067      | 0.8774     | 0.1707 | 24.9109       | 1.0014       | -23.9095 |


