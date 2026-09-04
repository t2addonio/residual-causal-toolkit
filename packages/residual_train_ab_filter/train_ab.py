#!/usr/bin/env python3
"""Arm A vs Arm B — residual filter in the train loop.
Arm A: CE only + FILTER at eval.
Arm B: CE + residual READ term on a contrastive direction refreshed in-loop.
Honest 2026-08-31 result on 4×5090: B is not ≥ A (success rule FAIL).
Sidecar used tau_sig=1.0 while observed p_sig is O(100–400), so B flattened the port.
Toolkit is not shown as a trainer on this task. Do not invent a third architecture.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

# Full GPU trainer lives in the working-store original (14k).
# This module records the locked protocol and the measured A/B outcome.

RESULT = {
    "box": "C.49450685 4x5090",
    "task": "gpt2 / mod-113 / layer 6 / 6 epochs / n_train=4096 / 8 seeds × 2 arms",
    "A": {"acc": 0.224, "acc_std": 0.045, "FILTER": "4/8", "p_sig": 245.5, "p_path": 108.1},
    "B": {"acc": 0.185, "acc_std": 0.050, "FILTER": "3/8", "p_sig": 17.73, "p_path": 8.94},
    "delta_B_minus_A": {"acc": -0.039, "FILTER": -1, "p_sig": -227.8},
    "success_rule": "FAIL — B is not ≥ A",
    "note": "both arms undertrained vs Phase D 36.2% baseline; sidecar tau_sig mis-scaled",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-json", type=str, default="")
    args = ap.parse_args()
    print(json.dumps(RESULT, indent=2))
    if args.write_json:
        Path(args.write_json).write_text(json.dumps(RESULT, indent=2))

if __name__ == "__main__":
    main()
