"""
Print Bias in Bios dataset statistics.
Uses the same split caps as run_all_baselines.py by default (10k train, 5k val, 10k test).
Run from project root: python scripts/run_bias_in_bios_stats.py [--bios-train-max 10000] [--bios-val-max 5000] [--bios-test-max 10000]
"""
import sys
import argparse
from pathlib import Path

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from data.loaders import get_qwen_tokenizer
from data.bias_in_bios import (
    load_bias_in_bios,
    get_bias_in_bios_stats,
    print_bias_in_bios_stats,
)

DEFAULT_BIOS_TRAIN_MAX = 10000
DEFAULT_BIOS_VAL_MAX = 5000
DEFAULT_BIOS_TEST_MAX = 10000


def main():
    parser = argparse.ArgumentParser(description="Bias in Bios stats (with same caps as run_all_baselines.py)")
    parser.add_argument("--bios-train-max", type=int, default=DEFAULT_BIOS_TRAIN_MAX,
                        help=f"Cap train split (default: {DEFAULT_BIOS_TRAIN_MAX})")
    parser.add_argument("--bios-val-max", type=int, default=DEFAULT_BIOS_VAL_MAX,
                        help=f"Cap val split (default: {DEFAULT_BIOS_VAL_MAX})")
    parser.add_argument("--bios-test-max", type=int, default=DEFAULT_BIOS_TEST_MAX,
                        help=f"Cap test split (default: {DEFAULT_BIOS_TEST_MAX})")
    args = parser.parse_args()

    tokenizer = get_qwen_tokenizer()
    print("Loading Bias in Bios (train/dev/test)...")
    train_ds, val_ds, test_ds = load_bias_in_bios(
        tokenizer,
        max_length=256,
        use_predefined_splits=True,
    )
    # Apply same caps as run_all_baselines.py
    if args.bios_train_max is not None and len(train_ds) > args.bios_train_max:
        train_ds = train_ds.select(range(args.bios_train_max))
    if args.bios_val_max is not None and len(val_ds) > args.bios_val_max:
        val_ds = val_ds.select(range(args.bios_val_max))
    if args.bios_test_max is not None and len(test_ds) > args.bios_test_max:
        test_ds = test_ds.select(range(args.bios_test_max))
    print(f"Using capped splits: Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    stats = get_bias_in_bios_stats(train_ds, val_ds, test_ds)
    print_bias_in_bios_stats(stats)


if __name__ == "__main__":
    main()
