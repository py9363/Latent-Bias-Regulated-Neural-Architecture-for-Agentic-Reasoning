"""
Plot Bias in Bios distributions (splits, gender, occupation) for README / reports.

Run from project root:
  python scripts/plot_bias_in_bios_data.py
  python scripts/plot_bias_in_bios_data.py --out results/figures --bios-train-max 5000

Writes PNGs under --out (default: results/figures).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.bias_in_bios import get_bias_in_bios_stats, load_bias_in_bios
from data.loaders import get_qwen_tokenizer

DEFAULT_OUT = root / "results" / "figures"


def _plot_split_sizes(stats: dict, out_path: Path) -> None:
    names = []
    vals = []
    for k, label in [("train_size", "Train"), ("val_size", "Val"), ("test_size", "Test")]:
        if k in stats:
            names.append(label)
            vals.append(stats[k])
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, vals, color=["#2c7bb6", "#abd9e9", "#fdae61"])
    ax.set_ylabel("Examples")
    ax.set_title("Bias in Bios — split sizes (after caps)")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.01, f"{v:,}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_gender_train(stats: dict, out_path: Path) -> None:
    gd = stats.get("train_gender_dist", {})
    if not gd:
        return
    labels = []
    counts = []
    for g in sorted(gd.keys()):
        labels.append("Male" if int(g) == 0 else "Female")
        counts.append(gd[g])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, counts, color=["#4daf4a", "#984ea3"])
    ax.set_ylabel("Count (train)")
    ax.set_title("Bias in Bios — gender distribution (train)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_occupation_train(stats: dict, out_path: Path) -> None:
    occ = stats.get("train_occupation_dist", {})
    if not occ:
        return
    pairs = sorted(occ.items(), key=lambda x: -x[1])
    ids = [p[0] for p in pairs]
    counts = [p[1] for p in pairs]
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(ids))
    ax.barh(y, counts, color="#3182bd")
    ax.set_yticks(y)
    ax.set_yticklabels([f"Occ {i}" for i in ids], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count (train)")
    ax.set_title("Bias in Bios — occupation counts (train, all 28 classes)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Bias in Bios distributions to PNG files")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory for PNGs")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="Tokenizer only (for loading)")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--bios-train-max", type=int, default=5000)
    p.add_argument("--bios-val-max", type=int, default=2500)
    p.add_argument("--bios-test-max", type=int, default=5000)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = get_qwen_tokenizer(args.model)
    train_ds, val_ds, test_ds = load_bias_in_bios(
        tokenizer, max_length=args.max_length, use_predefined_splits=True
    )
    if args.bios_train_max is not None and len(train_ds) > args.bios_train_max:
        train_ds = train_ds.select(range(args.bios_train_max))
    if args.bios_val_max is not None and len(val_ds) > args.bios_val_max:
        val_ds = val_ds.select(range(args.bios_val_max))
    if args.bios_test_max is not None and len(test_ds) > args.bios_test_max:
        test_ds = test_ds.select(range(args.bios_test_max))

    stats = get_bias_in_bios_stats(train_ds, val_ds, test_ds)

    _plot_split_sizes(stats, args.out / "bias_in_bios_split_sizes.png")
    _plot_gender_train(stats, args.out / "bias_in_bios_gender_train.png")
    _plot_occupation_train(stats, args.out / "bias_in_bios_occupation_train.png")

    print(f"Wrote PNGs to {args.out.resolve()}:")
    for name in (
        "bias_in_bios_split_sizes.png",
        "bias_in_bios_gender_train.png",
        "bias_in_bios_occupation_train.png",
    ):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
