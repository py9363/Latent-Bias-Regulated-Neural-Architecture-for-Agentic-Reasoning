#!/usr/bin/env python3
"""
Same pipeline as ``run_demo.py`` (B0–B3 + Main, CrowS/BBQ, agentic reports).

Default Bios caps only: **train 200, val 50, test 100**.
Pass ``--bios-train-max`` / ``--bios-val-max`` / ``--bios-test-max`` after other flags to override;
later values win.

From repo root::

  python demo/Short_Demo.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

_DEFAULT_BIOS_CAPS = [
    "--bios-train-max",
    "500",
    "--bios-val-max",
    "150",
    "--bios-test-max",
    "250",
]
_SKIP_EVAL_FLAGS = ["--crows-max", "0", "--bbq-max", "0"]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--skip-evals",
        action="store_true",
        help="Skip CrowS-Pairs and BBQ by forwarding --crows-max 0 --bbq-max 0.",
    )
    known, remaining = parser.parse_known_args(sys.argv[1:])

    forwarded = [sys.argv[0]] + _DEFAULT_BIOS_CAPS
    if known.skip_evals:
        forwarded += _SKIP_EVAL_FLAGS
    forwarded += remaining
    sys.argv = forwarded

    demo_dir = Path(__file__).resolve().parent
    run_demo_path = demo_dir / "run_demo.py"
    spec = importlib.util.spec_from_file_location("run_demo", run_demo_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    main()
