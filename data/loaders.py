"""
Dataset loaders for bias evaluation.
- CrowS-Pairs: paired sentences + stereotypical preference (evaluation only)
- BBQ: question, context, answers, correct answer, protected attribute (evaluation only)
Training/probing uses Bias in Bios via data.bias_in_bios.
"""
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from datasets import load_dataset, Dataset, concatenate_datasets as hf_concat
from transformers import AutoTokenizer


def get_qwen_tokenizer(model_name: str = "Qwen/Qwen2.5-0.5B"):
    """Load Qwen tokenizer for consistent tokenization across all datasets."""
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


# ---------------------------------------------------------------------------
# CrowS-Pairs
# ---------------------------------------------------------------------------

CROWS_PAIRS_HF = "nyu-mll/crows_pairs"
CROWS_PAIRS_CSV_URL = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"

def _load_crows_pairs_from_csv() -> Dataset:
    """Load CrowS-Pairs from official CSV (fallback when HF dataset scripts are disabled)."""
    import csv
    import urllib.request
    import io
    req = urllib.request.urlopen(CROWS_PAIRS_CSV_URL, timeout=30)
    text = req.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    # First column is unnamed in CSV (header ",sent_more,...") so DictReader gives key ""
    id_key = "" if "" in (rows[0] if rows else {}) else "id"
    ids = [int(r.get(id_key, i)) for i, r in enumerate(rows)]
    sent_more = [r["sent_more"] for r in rows]
    sent_less = [r["sent_less"] for r in rows]
    stereo_antistereo = [str(r.get("stereo_antistereo", "stereo")).strip().lower() for r in rows]
    bias_type_list = [str(r.get("bias_type", "")).strip() for r in rows]
    return Dataset.from_dict({
        "id": ids,
        "sent_more": sent_more,
        "sent_less": sent_less,
        "stereo_antistereo": stereo_antistereo,
        "bias_type": bias_type_list,
    })


def load_crows_pairs(
    split: Optional[str] = None,
    bias_type: Optional[str] = None,
) -> Dataset:
    """
    Load CrowS-Pairs (from HuggingFace or CSV fallback).
    Returns dataset with paired sentences and stereotypical preference.

    Each example has:
    - sent_more: sentence rated "more" stereotypical in the pair
    - sent_less: sentence rated "less" stereotypical
    - stereo_antistereo: "stereo" if sent_more is the stereotypical one, else "antistereo"
    - stereotypical_sentence: the sentence that is stereotypical (for convenience)
    - anti_stereotypical_sentence: the other sentence
    - bias_type: category (race-color, gender, etc.)
    """
    try:
        dataset = load_dataset(
            CROWS_PAIRS_HF,
            split="test" if split is None else split,
        )
    except Exception as e:
        err = str(e).lower()
        if (
            "scripts are no longer supported" in err
            or "dataset script" in err
            or "trust_remote_code" in err
            or "custom code" in err
            or "not supported anymore" in err
        ):
            dataset = _load_crows_pairs_from_csv()
        else:
            raise

    def add_preference(ex):
        st = ex.get("stereo_antistereo")
        if isinstance(st, str):
            is_stereo_more = st == "stereo"
        else:
            is_stereo_more = st == 0
        stereo_sent = ex["sent_more"] if is_stereo_more else ex["sent_less"]
        anti_sent = ex["sent_less"] if is_stereo_more else ex["sent_more"]
        return {
            **ex,
            "stereotypical_sentence": stereo_sent,
            "anti_stereotypical_sentence": anti_sent,
            "prefer_stereotypical": None,
        }

    dataset = dataset.map(add_preference, desc="CrowS-Pairs preference")
    if bias_type is not None:
        dataset = dataset.filter(lambda x: x.get("bias_type") == bias_type)
    return dataset


# ---------------------------------------------------------------------------
# BBQ
# ---------------------------------------------------------------------------

BBQ_HF = "HiTZ/bbq"
BBQ_CONFIGS = [
    "Age_ambig", "Age_disambig",
    "Disability_status_ambig", "Disability_status_disambig",
    "Gender_identity_ambig", "Gender_identity_disambig",
    "Nationality_ambig", "Nationality_disambig",
    "Physical_appearance_ambig", "Physical_appearance_disambig",
    "Race_ethnicity_ambig", "Race_ethnicity_disambig",
    "Religion_ambig", "Religion_disambig",
    "SES_ambig", "SES_disambig",
    "Sexual_orientation_ambig", "Sexual_orientation_disambig",
]


def load_bbq(
    config: Optional[str] = None,
    split: str = "test",
    configs: Optional[List[str]] = None,
) -> Dataset:
    """
    Load BBQ (Bias Benchmark for QA) from HuggingFace.

    Returns examples with:
    - question: str
    - context: str
    - answer options: ans0, ans1, ans2 (or list)
    - correct_answer: index 0/1/2 (label)
    - protected_attribute: category (e.g. Race_ethnicity, Gender_identity)
    - context_condition: ambig / disambig
    """
    if config is not None:
        dataset = load_dataset(BBQ_HF, config, split=split)
        dataset = dataset.add_column("protected_attribute", [config.split("_")[0] if "_" in config else config] * len(dataset))
        return dataset

    if configs is None:
        configs = BBQ_CONFIGS
    all_ds = []
    for cfg in configs:
        try:
            ds = load_dataset(BBQ_HF, cfg, split=split)
            attr = cfg.rsplit("_", 1)[0] if "_" in cfg else cfg
            ds = ds.add_column("protected_attribute", [attr] * len(ds))
            all_ds.append(ds)
        except Exception:
            continue
    if not all_ds:
        raise ValueError("No BBQ configs could be loaded.")
    return _concat(all_ds)


def _concat(datasets: List[Dataset]) -> Dataset:
    return hf_concat(datasets)


def bbq_example_to_dict(example: Dict) -> Dict[str, Any]:
    """Normalize one BBQ example to question, context, options, correct index, protected attribute."""
    return {
        "question": example["question"],
        "context": example["context"],
        "ans0": example.get("ans0", ""),
        "ans1": example.get("ans1", ""),
        "ans2": example.get("ans2", ""),
        "answer_options": [example.get("ans0", ""), example.get("ans1", ""), example.get("ans2", "")],
        "correct_answer": int(example["label"]),
        "protected_attribute": example.get("protected_attribute", example.get("category", "unknown")),
        "context_condition": example.get("context_condition", "disambig"),
    }


