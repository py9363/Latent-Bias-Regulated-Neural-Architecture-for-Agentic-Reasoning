# Mitigating Gender Bias in Occupation Classification from Biographies

**Author:** Pooja Yakkala  
**Course / milestone:** Capstone (Checkpoint 3)

I study **gender bias** in language models fine-tuned for **occupation prediction** from biographies. This repository is my full implementation: **Bias in Bios** training and evaluation, **CrowS-Pairs** and **BBQ** benchmarks, several debiasing baselines, a **stability-regularized adversarial Main** model built on **Qwen2.5** with **LoRA**, and an **agentic** evaluation stack (multi-step adaptation, biography probes, LM-style task shift).

---

## What this project implements

1. **Baselines:** **B0** (pretrained, no fine-tuning), **B1** (standard cross-entropy fine-tuning), **B2** (adversarial debiasing with a gradient-reversal bias head and LoRA), **B3** (INLP / iterative null-space projection), and **Main** (same adversarial setup as B2 plus a **stability** term on the bias head under **simulated task-only inner updates**).
2. **Main objective:** I train Main with  
   \(\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{bias}} + \lambda_2 \mathcal{L}_{\text{stab}}\),  
   where \(\mathcal{L}_{\text{stab}}\) compares bias-head logits **before** and **after** a short inner loop that updates LoRA and the task head on a **model twin** while keeping the bias head fixed (KL stability by default). Implementation: `baselines/main_stability.py`.
3. **Evaluation:** occupation accuracy and gender gap on Bios; **linear probe** recoverability **\(R(\theta)\)** on pooled representations; optional **LoRA adaptation** from each checkpoint to report **\(R(\theta')\)** and **\(\Delta R\)**; CrowS-Pairs and BBQ scores; **agentic** tables (E-step notation, biography fine-tuning, TABLE 0–5) via `run_agentic_baselines.py`.

---

## Framework (high-level)

```mermaid
flowchart TB
  subgraph data [Data]
    BIO[Bias in Bios HF]
    CP[CrowS-Pairs eval]
    BBQ[BBQ eval]
  end

  subgraph train [Training baselines]
    B1[B1 Standard FT]
    B2[B2 Adversarial GRL]
    B3[B3 INLP]
    M[Main L_task + λ1 L_bias + λ2 L_stab + LoRA]
  end

  subgraph eval [Evaluation]
    PROBE[Linear gender probe on pooled h]
    AGENT[Agentic prompts + inner adaptation]
    LM[LM task-shift: Summarize biography]
  end

  BIO --> B1 & B2 & B3 & M
  B1 & B2 & B3 & M --> PROBE
  B1 & B2 & B3 & M --> AGENT
  AGENT --> LM
```

**Figure 1.** Bios (and optional eval sets) feed the baseline trainers; outputs support static probes, agentic multi-step evaluation, and optional LM-style adaptation on biographies.

---

## Repository structure

| Path | Purpose |
|------|---------|
| `baselines/` | `b1_standard.py`, `b2_adversarial.py`, `b3_inlp.py`, `main_stability.py` |
| `models/` | `qwen_task.py`, `adversarial.py`, `grl.py` |
| `data/` | `bias_in_bios.py`, `loaders.py`, `adaptation_labels.py`, `samples/` |
| `evaluation/` | `probe.py`, `metrics.py`, `capstone_report_md.py`, `lm_summarize_adapt.py` |
| `adaptation/` | `lora_adaptation.py` — short LoRA fine-tune and \(\Delta R\) |
| `demo/` | `run_demo.py`, `README.md` — smaller end-to-end path |
| `scripts/` | `plot_bias_in_bios_data.py` |
| `config.py` | Global defaults (LR, \(\lambda\), LoRA, inner steps, device, paths) |
| `run_all_baselines.py` | Primary entry: B0–B3 + Main + CrowS/BBQ + LoRA \(\Delta R\) + agentic bundle |
| `run_agentic_baselines.py` | Agentic library (imported by `run_all_baselines.py` and `demo/run_demo.py`) |
| `results/` | JSON/Markdown reports and `figures/` for dataset plots |

---

## Environment

**Requirements:** Python 3.10+ recommended; CUDA optional (CPU is possible but slow).

```bash
pip install -r requirements.txt
```

Core stack: `torch`, `transformers`, `datasets`, `peft`, `scikit-learn`, `numpy`, `accelerate`, `tqdm`.

**Device:** I select the GPU index in `config.py` (`CUDA_DEVICE_ID`). If CUDA is unavailable, training falls back to CPU. **Results** write to `results/` by default, or to an external results folder when that path exists (see `RESULTS_DIR` in `config.py`).

---

## Data

| Dataset | Source | Role |
|---------|--------|------|
| **Bias in Bios** | Hugging Face `LabHC/bias_in_bios` | Train / val / test for occupation + probing |
| **CrowS-Pairs** | `nyu-mll/crows_pairs` | Stereotype preference (eval) |
| **BBQ** | `HiTZ/bbq` | Disambiguated QA gap (eval) |

Loaders cache splits on first use. For schema without downloading the full corpus, see `data/samples/bias_in_bios_example.json` and `data/samples/README.md`.

**Figures** (example plots): `results/figures/gender_distribution.png`, `results/figures/occupation_distribution.png`. Regenerate with:
**Dataset statistics (optional):**

```bash
python scripts/run_bias_in_bios_stats.py
```

### Dataset distributions (figures)

Committed plots under **`results/figures/`** (Bias in Bios, predefined splits; counts depend on caps when the script ran).

**Gender (train)** — `results/figures/gender_distribution.png`

<p align="center">
  <img src="./results/figures/gender_distribution.png" alt="Bias in Bios: gender distribution on train split" width="520" />
</p>

**Occupation (train, 28 classes)** — `results/figures/occupation_distribution.png`

<p align="center">
  <img src="./results/figures/occupation_distribution.png" alt="Bias in Bios: occupation counts on train split" width="720" />
</p>

**Regenerate plots** (default output names: `bias_in_bios_split_sizes.png`, `bias_in_bios_gender_train.png`, `bias_in_bios_occupation_train.png` in `results/figures/`):

```bash
python scripts/plot_bias_in_bios_data.py
python scripts/plot_bias_in_bios_data.py --out results/figures --bios-train-max 5000
```

---

## Full baseline run (`run_all_baselines.py`)

```bash
python run_all_baselines.py
```

**Default Bias in Bios caps** (CLI: `--bios-train-max`, `--bios-val-max`, `--bios-test-max`): **5000 / 1250 / 2500** examples. **CrowS-Pairs** default **500** rows (`--crows-max`); **BBQ** default **2000** (`--bbq-max`). **Sequence length** default **256** (`--max-length`). **Epochs** default from `config.py` (**3**). **Batch size** defaults to `max(16, DEFAULT_BATCH_SIZE)` (typically **16**). **Seed** default **42**.

**B2 and Main (LoRA)** use `config.py` defaults unless overridden: **rank 32**, **alpha 64**, dropout **0.1**. For the full sweep I pin **run-all-specific** hyperparameters in `run_all_baselines.py` (they apply to B2’s \(\lambda_1\) and Main’s full objective): default **`--lambda-bias` 0.5**, stability weight \(\lambda_2\) **0.05**, Main **inner learning rate** **5e-5**, **gradient clipping** **0.5**, and **inner steps** from `DEFAULT_MAIN_INNER_STEPS` in `config.py` (**4** per batch). I enable **bias loss balancing** unless I pass `--no-bias-loss-balance`.

**Other useful flags:** `--quick` (caps CrowS/BBQ to 200 and shortens epochs), `--model`, `--lora-r`, `--lora-alpha`, `--agentic-models b_adv` (default) or `all` for extra agentic models in JSON.

**Outputs:** timestamped `report_YYYYMMDD_HHMMSS.json` / `.md` under `RESULTS_DIR`, plus stable copies **`run_all_full_report.json`** and **`run_all_report.md`**, and **`baseline_comparison.json`**. The Markdown report includes the baseline metrics table and the agentic section when agentic evaluation completes.

**Agentic phase** (after training) uses its own adaptation settings documented in `meta` inside the JSON (steps, LR, etc.); those differ from Main’s *training* inner loop.

---

## Demo (smaller run)

```bash
python demo/run_demo.py
```

Options and outputs are described in `demo/README.md` (`--cpu`, `--skip-main`, `--full-debias`, checkpoints directory, etc.).

---

## Results I report

Numbers depend on hardware, caps, and hyperparameters. I treat **`results/run_all_full_report.json`** (and matching `run_all_report.md`) as the canonical artifact for a full sweep; **`baseline_comparison.json`** summarizes occupation accuracy, gaps, \(R(\theta)\), LoRA \(\Delta R\), CrowS, and BBQ in one place.

**Example agentic snapshot** (fixed split / older run): see `results/agentic_report_20260403_122754.json` and `.md` for TABLE 0–5 style metrics. **Demo** outputs live under `demo/output/` after `demo/run_demo.py`.

**Claims I defend in writeups:** debiasing lowers probe recoverability relative to standard fine-tuning; **Main** targets **stable** bias-head behavior under simulated task adaptation; CrowS/BBQ and agentic tables provide complementary views of bias under static and adaptive evaluation.

---

## Demo and presentation pointers

- **Data:** `data/bias_in_bios.py`, `data/loaders.py`; distributions via `scripts/plot_bias_in_bios_data.py`.
- **Models:** `models/adversarial.py`, `baselines/main_stability.py` for the Main training loop and stability term.
- **Metrics:** `evaluation/probe.py`, `evaluation/metrics.py`; reports rendered by `evaluation/capstone_report_md.py`.

---

## References

- **Bias in Bios:** De-Arteaga et al., *Bias in Bios: A Case for Intersectional Fairness* ([dataset](https://huggingface.co/datasets/LabHC/bias_in_bios)).
- **INLP:** Ravfogel et al., *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection*.
- **Gradient reversal / adversarial fairness:** Ganin et al., *Domain-Adversarial Training of Neural Networks*; fair-representation literature.
- **CrowS-Pairs:** Nangia et al.; **BBQ:** Parrish et al. (Hugging Face dataset IDs match `data/loaders.py`).
- **Backbone:** [Qwen2.5](https://huggingface.co/Qwen); **LoRA:** Hugging Face `peft`.

---

## License

Project source code in this repository is under the [MIT License](LICENSE).

**Third-party:** pretrained weights (e.g. Qwen2.5), Python packages, and datasets each carry their own licenses (model cards, PyPI, Hugging Face dataset pages).
