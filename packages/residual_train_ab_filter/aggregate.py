#!/usr/bin/env python3
"""Aggregate Arm A vs Arm B JSON seeds into one table."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/ab_summary.json")
    ap.add_argument("--md", default="results/ab_summary.md")
    args = ap.parse_args()
    root = Path(args.results)
    rows = [json.loads(p.read_text()) for p in sorted(root.glob("arm*_seed*.json"))]
    if not rows:
        raise SystemExit(f"no arm*_seed*.json under {root}")
    summary = {"n_files": len(rows), "arms": {}}
    lines = ["# Train A/B — residual filter in the loop", "",
             "| arm | n | mean acc | FILTER pass/n | mean p_sig | mean p_path | mean epochs-pass |",
             "|-----|---|----------|---------------|------------|-------------|------------------|"]
    for a in ("A", "B"):
        rs = [r for r in rows if r.get("arm") == a]
        if not rs:
            continue
        acc = [r.get("final_val_acc") or 0.0 for r in rs]
        passed = [1 if r.get("final_filter_passed") else 0 for r in rs]
        ps = [r.get("final_p_sig") or 0.0 for r in rs]
        pp = [r.get("final_p_path") or 0.0 for r in rs]
        ep = [r.get("n_epochs_pass") or 0 for r in rs]
        rec = {"n": len(rs), "mean_acc": float(np.mean(acc)), "std_acc": float(np.std(acc)),
               "n_filter_pass": int(sum(passed)), "mean_p_sig": float(np.mean(ps)),
               "mean_p_path": float(np.mean(pp)), "mean_epochs_pass": float(np.mean(ep)),
               "seeds": [r["seed"] for r in rs]}
        summary["arms"][a] = rec
        lines.append(f"| {a} | {rec['n']} | {rec['mean_acc']:.3f}±{rec['std_acc']:.3f} | {rec['n_filter_pass']}/{rec['n']} | {rec['mean_p_sig']:.4f} | {rec['mean_p_path']:.4f} | {rec['mean_epochs_pass']:.2f} |")
    if set(summary["arms"]) >= {"A", "B"}:
        A, B = summary["arms"]["A"], summary["arms"]["B"]
        lines += ["", f"Delta B-A acc {B['mean_acc']-A['mean_acc']:+.3f} FILTER {B['n_filter_pass']-A['n_filter_pass']:+d} p_sig {B['mean_p_sig']-A['mean_p_sig']:+.1f}"]
    Path(args.md).write_text("\n".join(lines) + "\n")
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))

if __name__ == "__main__":
    main()
