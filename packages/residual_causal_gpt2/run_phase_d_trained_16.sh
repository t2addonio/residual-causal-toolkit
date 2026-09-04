#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"}
GPUS=${GPUS:-"0 1 2 3"}
LAYER=${LAYER:-"6"}
OUT=${OUT:-"results/phase_d_trained"}
EPOCHS=${EPOCHS:-"4"}
N_TRAIN=${N_TRAIN:-"2048"}
mkdir -p "$OUT" logs
GPU_ARR=($GPUS)
N_GPU=${#GPU_ARR[@]}
i=0
for SEED in $SEEDS; do
  GPU=${GPU_ARR[$((i % N_GPU))]}
  LOG="logs/phase_d_trained_seed${SEED}_gpu${GPU}.log"
  echo "[launch] Phase D trained seed=$SEED gpu=$GPU → $LOG"
  CUDA_VISIBLE_DEVICES=$GPU python phase_d_trained.py \
    --seed "$SEED" --device "cuda:0" --layer "$LAYER" \
    --epochs "$EPOCHS" --n-train "$N_TRAIN" --out-dir "$OUT" > "$LOG" 2>&1 &
  i=$((i + 1))
  if (( i % N_GPU == 0 )); then wait; fi
done
wait
echo "[done] Phase D trained 16-seed complete. Results in $OUT/"
