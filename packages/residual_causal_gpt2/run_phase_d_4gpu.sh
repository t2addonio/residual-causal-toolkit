#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
GPUS=${GPUS:-"0 1 2 3"}
LAYER=${LAYER:-"6"}
OUT=${OUT:-"results/phase_d"}
mkdir -p "$OUT" logs
GPU_ARR=($GPUS)
N_GPU=${#GPU_ARR[@]}
i=0
for SEED in $SEEDS; do
  GPU=${GPU_ARR[$((i % N_GPU))]}
  LOG="logs/phase_d_seed${SEED}_gpu${GPU}.log"
  echo "[launch] Phase D seed=$SEED gpu=$GPU → $LOG"
  CUDA_VISIBLE_DEVICES=$GPU python phase_d_online_residual.py \
    --seed "$SEED" --device "cuda:0" --layer "$LAYER" --out-dir "$OUT" \
    > "$LOG" 2>&1 &
  i=$((i + 1))
  if (( i % N_GPU == 0 )); then wait; fi
done
wait
echo "[done] Phase D 4-GPU multi-seed complete. Results in $OUT/"
