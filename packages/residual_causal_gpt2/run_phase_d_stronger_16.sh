#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"}
GPUS=${GPUS:-"0 1 2 3"}
LAYER=${LAYER:-"6"}
OUT=${OUT:-"results/phase_d_stronger"}
N_TRAIN=${N_TRAIN:-"8192"}
EPOCHS=${EPOCHS:-"12"}
mkdir -p "$OUT" logs
GPU_ARR=($GPUS)
N_GPU=${#GPU_ARR[@]}
i=0
for SEED in $SEEDS; do
  GPU=${GPU_ARR[$((i % N_GPU))]}
  LOG="logs/phase_d_stronger_seed${SEED}_gpu${GPU}.log"
  echo "[launch] Phase D-stronger seed=$SEED gpu=$GPU → $LOG"
  CUDA_VISIBLE_DEVICES=$GPU python phase_d_stronger.py \
    --seed "$SEED" --device "cuda:0" --layer "$LAYER" \
    --n-train "$N_TRAIN" --epochs "$EPOCHS" --out-dir "$OUT" > "$LOG" 2>&1 &
  i=$((i + 1))
  if (( i % N_GPU == 0 )); then wait; fi
done
wait
echo "[done] Phase D-stronger 16-seed complete. Results in $OUT/"
