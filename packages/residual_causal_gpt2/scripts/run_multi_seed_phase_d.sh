#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
GPUS=${GPUS:-"0"}
MODEL=${MODEL:-"gpt2"}
LAYER=${LAYER:-"6"}
OUT=${OUT:-"results/phase_d"}
N_SIG=${N_SIG:-"64"}
N_HELD=${N_HELD:-"64"}
mkdir -p "$OUT" logs
GPU_ARR=($GPUS)
N_GPU=${#GPU_ARR[@]}
i=0
for SEED in $SEEDS; do
  GPU=${GPU_ARR[$((i % N_GPU))]}
  LOG="logs/phase_d_seed${SEED}_gpu${GPU}.log"
  echo "[launch] Phase D online residual seed=$SEED gpu=$GPU → $LOG"
  CUDA_VISIBLE_DEVICES=$GPU python src/phase_d_online_residual_control.py \
    --seed "$SEED" --device "cuda:0" --model "$MODEL" --layer "$LAYER" \
    --n-sig "$N_SIG" --n-held "$N_HELD" --out-dir "$OUT" > "$LOG" 2>&1 &
  i=$((i + 1))
  if (( i % N_GPU == 0 )); then wait; fi
done
wait
echo "[done] Phase D multi-seed online residual control complete."
