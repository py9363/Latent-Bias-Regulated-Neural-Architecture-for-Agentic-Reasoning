"""
Bias in Bios: biography text (x), occupation label (y), gender label (s).
Primary training and probing dataset for studying gender recoverability and drift under LoRA.
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from collections import Counter

from datasets import load_dataset, Dataset


BIAS_IN_BIOS_HF = "LabHC/bias_in_bios"
NUM_OCCUPATIONS = 28
GENDER_MAP = {"male": 0, "female": 1, "Male": 0, "Female": 1, 0: 0, 1: 1}


def _to_int(x: Any) -> int:
    if isinstance(x, int):
        return int(x)
    if isinstance(x, str):
        return int(x.strip()) if x.strip().isdigit() else 0
    return int(x)


def load_bias_in_bios(
    tokenizer,
    split: Optional[str] = None,
    max_length: int = 256,
    text_column: str = "hard_text",
    occupation_column: str = "profession",
    gender_column: str = "gender",
    use_predefined_splits: bool = True,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load Bias in Bios. Returns (train_ds, val_ds, test_ds) with:
    - input_ids, attention_mask
    - occupation_label (0..27)
    - gender_label (0/1)
    - label (alias for occupation_label for compatibility)
    - sensitive_attribute (alias for gender_label for compatibility)
    """
    if use_predefined_splits:
        train_raw = load_dataset(BIAS_IN_BIOS_HF, split="train")
        dev_raw = load_dataset(BIAS_IN_BIOS_HF, split="dev")
        test_raw = load_dataset(BIAS_IN_BIOS_HF, split="test")
        splits = [train_raw, dev_raw, test_raw]
    else:
        full = load_dataset(BIAS_IN_BIOS_HF, split="train")
        n = len(full)
        np.random.seed(seed)
        idx = np.random.permutation(n)
        t1 = int(n * train_fraction)
        t2 = int(n * (train_fraction + val_fraction))
        splits = [
            full.select(idx[:t1]),
            full.select(idx[t1:t2]),
            full.select(idx[t2:]),
        ]

    def _parse_occupation(v):
        if isinstance(v, (int, float)):
            return int(v)
        return hash(str(v)) % NUM_OCCUPATIONS

    def _process_and_tokenize(ds):
        cols = ds.column_names if hasattr(ds, "column_names") else list(ds[0].keys()) if len(ds) else []
        text_col = text_column if text_column in cols else ("text" if "text" in cols else cols[0] if cols else "hard_text")
        occ_col = occupation_column if occupation_column in cols else "profession"
        gen_col = gender_column if gender_column in cols else "gender"
        texts = []
        occ_labels = []
        gender_labels = []
        for i in range(len(ds)):
            row = ds[i]
            text = row.get(text_col) or row.get("hard_text") or row.get("text", "")
            occ = row.get(occ_col, 0)
            g = row.get(gen_col, 0)
            occ_int = _to_int(occ) if isinstance(occ, (int, float)) else _parse_occupation(occ)
            g_int = GENDER_MAP.get(g, g)
            if isinstance(g_int, str):
                g_int = GENDER_MAP.get(g_int.lower(), 0)
            g_int = _to_int(g_int)
            if g_int not in (0, 1):
                g_int = 0
            occ_int = max(0, min(occ_int, NUM_OCCUPATIONS - 1))
            texts.append(text)
            occ_labels.append(occ_int)
            gender_labels.append(g_int)
        enc = tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors=None,
        )
        enc["occupation_label"] = occ_labels
        enc["gender_label"] = gender_labels
        enc["label"] = occ_labels
        enc["sensitive_attribute"] = gender_labels
        return Dataset.from_dict(enc)

    out = []
    for s in splits:
        out.append(_process_and_tokenize(s))
    return tuple(out)


def get_bias_in_bios_stats(
    train_ds: Dataset,
    val_ds: Optional[Dataset] = None,
    test_ds: Optional[Dataset] = None,
) -> Dict[str, Any]:
    """
    Compute dataset statistics:
    - total size (per split)
    - occupation distribution
    - gender distribution
    - gender imbalance per occupation
    """
    def _occ_key(ds, i):
        return ds[i].get("occupation_label", ds[i].get("label", 0))

    def _gender_key(ds, i):
        return ds[i].get("gender_label", ds[i].get("sensitive_attribute", 0))

    stats = {}
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if ds is None:
            continue
        n = len(ds)
        stats[f"{name}_size"] = n
        occ_counts = Counter(_occ_key(ds, i) for i in range(n))
        gender_counts = Counter(_gender_key(ds, i) for i in range(n))
        stats[f"{name}_occupation_dist"] = dict(occ_counts)
        stats[f"{name}_gender_dist"] = dict(gender_counts)

    # Gender imbalance per occupation (on train)
    n_train = len(train_ds)
    occ_to_genders = {}
    for i in range(n_train):
        o = _occ_key(train_ds, i)
        g = _gender_key(train_ds, i)
        occ_to_genders.setdefault(o, []).append(g)
    imbalance = {}
    for o, genders in occ_to_genders.items():
        c = Counter(genders)
        total = len(genders)
        if total == 0:
            imbalance[o] = 0.0
        else:
            pct_female = 100.0 * c.get(1, 0) / total
            imbalance[o] = round(pct_female, 2)
    stats["gender_imbalance_per_occupation"] = imbalance
    stats["total_size"] = n_train + (len(val_ds) if val_ds else 0) + (len(test_ds) if test_ds else 0)
    return stats


def print_bias_in_bios_stats(stats: Dict[str, Any]) -> None:
    """Print stats to stdout."""
    print("Bias in Bios — Dataset statistics")
    print("=" * 50)
    print("Total size:", stats.get("total_size", "N/A"))
    for k in ["train_size", "val_size", "test_size"]:
        if k in stats:
            print(f"  {k}: {stats[k]}")
    print("\nOccupation distribution (train):")
    occ = stats.get("train_occupation_dist", {})
    for o, c in sorted(occ.items(), key=lambda x: -x[1])[:15]:
        print(f"  occupation {o}: {c}")
    if len(occ) > 15:
        print(f"  ... and {len(occ) - 15} more")
    print("\nGender distribution (train):")
    for g, c in sorted(stats.get("train_gender_dist", {}).items()):
        print(f"  {g}: {c}")
    print("\nGender imbalance per occupation (% female):")
    imb = stats.get("gender_imbalance_per_occupation", {})
    for o in sorted(imb.keys())[:10]:
        print(f"  occupation {o}: {imb[o]}%")
    if len(imb) > 10:
        print(f"  ... and {len(imb) - 10} more")
