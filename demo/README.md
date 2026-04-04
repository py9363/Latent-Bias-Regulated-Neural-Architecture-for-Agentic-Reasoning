# Bias in Bios — full baseline demo

Uses the same `load_bias_in_bios` pipeline as the main project, but on a **tiny shared train / val / test split** (defaults: 96 train / 24 val / 32 test, **5 epochs** per trainable stage). It runs the same **B0 → B1 → B2 → B3 → Main** sequence as `run_all_baselines.py`, **without** CrowS-Pairs or BBQ.

| Step | What runs |
|------|-----------|
| **B0** | Pretrained `QwenTaskModel`: task accuracy + linear gender probe on pooled hidden states (test) |
| **B1** | Standard fine-tune |
| **B2** | Adversarial debiasing + LoRA |
| **B3** | INLP (few iterations by default) |
| **Main** | Stability-regularized adversarial (`run_main`; inner loop can be slow on CPU) |

## Run (from repository root)

```bash
pip install -r requirements.txt
python demo/run_demo.py
```

CPU only:

```bash
python demo/run_demo.py --cpu
```

Faster / smaller:

```bash
python demo/run_demo.py --train-samples 48 --val-samples 12 --test-samples 24 --epochs 1 --inlp-k 2 --batch-size 4
```

Skip Main (e.g. slow CPU inner loop):

```bash
python demo/run_demo.py --skip-main
```

**Tiny data note:** By default the demo uses **milder** `lambda_bias` / `lambda_stab`, **fewer** Main inner steps, a **capped** INLP `k`, and **at least 3 B3 epochs** when `n_train < 200` and you pass fewer than 3 global epochs, so B2/B3/Main keep non-trivial occupation accuracy. To match full `run_all_baselines.py` debiasing strengths (often collapses task acc on ~100 examples), run:

```bash
python demo/run_demo.py --full-debias
```

## Outputs

| File | Contents |
|------|----------|
| `demo/output/demo_log.txt` | Console log (tee) |
| `demo/output/demo_full_report.json` | Meta, bios-split metrics, optional agentic block, paths |
| `demo/output/demo_agentic_report.md` | Same TABLE 0–5 structure as full `run_agentic_baselines.py` (tiny split) |
| `demo/output/demo_agentic_report.json` | Machine-readable agentic metrics |
| `demo/output/checkpoints/{b1,b2,b3,main}_demo/` | Checkpoints from each trained stage |
| `demo/data/sample_batch.json` | First few examples: text preview, occupation id, gender label |

Use `--skip-agentic-report` to skip agentic eval; `--skip-table0` to skip TABLE 0 only (less memory).

**Task shift (default):** TABLE 0 and the agentic inner loop use **causal LM** on the fixed instruction **"Summarize this biography."** plus the biography and a short **pseudo-summary** target (first sentence / truncated text) — not occupation classification — so the backbone shifts under a **generation-style** objective while the **gender** linear probe on pooled states is unchanged. Use `--adapt-objective occupation` for the same 28-way head as debiasing. Tune sequence length with `--adapt-lm-max-length` (default 512).

First run downloads **Qwen** weights and the **Bias in Bios** dataset from Hugging Face (network required).

## Relation to `run_all_baselines.py`

`run_all_baselines.py` trains on full (or `--quick`) data with extra evals. This script is a **miniature** version for smoke tests, screenshots, and local verification of the full pipeline.
