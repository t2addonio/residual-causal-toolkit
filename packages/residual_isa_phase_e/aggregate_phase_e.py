#!/usr/bin/env python3
"""Aggregate Phase E interferometer JSON results across seeds."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/phase_e")
    ap.add_argument("--out", type=str, default="results/phase_e/phase_e_aggregate.json")
    args = ap.parse_args()
    root = Path(args.root)
    files = sorted(root.glob("phase_e_seed*_layer*.json"))
    rows = [json.loads(f.read_text()) for f in files]
    if not rows:
        print("[warn] no Phase E results found"); return
    checks = [r["checks"] for r in rows]
    agg = {
        "n_seeds": len(rows),
        "seed_pass_rate": float(np.mean([c["seed_pass"] for c in checks])),
        "unitary_rate": float(np.mean([c["unitary_plane"] for c in checks])),
        "core_gt_random_rate": float(np.mean([c["core_fringe_gt_random"] for c in checks])),
        "zero_collapses_rate": float(np.mean([c["zero_collapses"] for c in checks])),
        "core_logprob_ptp_mean": float(np.mean([c["core_logprob_ptp"] for c in checks])),
        "rand_logprob_ptp_mean": float(np.mean([c["rand_logprob_ptp"] for c in checks])),
        "seeds": [r["seed"] for r in rows],
        "passed_seeds": [r["seed"] for r, c in zip(rows, checks) if c["seed_pass"]],
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=2))

if __name__ == "__main__":
    main()
