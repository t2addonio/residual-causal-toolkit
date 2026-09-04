#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DEV=${1:-cpu}
mkdir -p results/smoke logs
python -u train_ab.py --arm A --seed 0 --device "$DEV" --epochs 1 --n-train 128 --n-probe 8 --batch-size 8 --out-dir results/smoke
python -u train_ab.py --arm B --seed 0 --device "$DEV" --epochs 1 --n-train 128 --n-probe 8 --batch-size 8 --out-dir results/smoke
python aggregate.py --results results/smoke --out results/smoke/ab_summary.json --md results/smoke/ab_summary.md
echo "[smoke] ok"
