#!/usr/bin/env python3
"""Aggregate Phase D online residual control JSON results."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results/phase_d")
    ap.add_argument("--out", type=str, default="results/phase_d/phase_d_aggregate.json")
    args = ap.parse_args()
    files = sorted(Path(args.root).glob("phase_d_seed*_layer*.json"))
    rows = [json.loads(f.read_text()) for f in files]
    if not rows:
        print("[warn] no Phase D results found"); return
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_seeds": len(rows), "seeds": [r.get("seed") for r in rows]}, indent=2))
    print(f"[done] {len(rows)} seeds → {out}")

if __name__ == "__main__":
    main()
