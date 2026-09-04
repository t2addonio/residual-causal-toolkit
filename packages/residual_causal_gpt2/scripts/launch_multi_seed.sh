#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
GPUS=${GPUS:-"0"}
STEPS=${STEPS:-3000}
OUT=${OUT:-runs/train}
mkdir -p "$OUT" logs
GPU_ARR=($GPUS)
N_GPU=${#GPU_ARR[@]}
i=0
for seed in $SEEDS; do
  gpu=${GPU_ARR[$((i % N_GPU))]}
  echo "[launch] seed=$seed on cuda:$gpu"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/train_one_seed.py \
    --seed "$seed" --out_dir "$OUT" --steps "$STEPS" --device cuda \
    > "logs/train_seed_${seed}.log" 2>&1 &
  i=$((i + 1))
  if (( i % N_GPU == 0 )); then wait; fi
done
wait
echo "[launch] DONE"
