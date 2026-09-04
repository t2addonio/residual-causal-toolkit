#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2 3"}
GPUS=${GPUS:-"0"}
MODEL=${MODEL:-"gpt2"}
LAYER=${LAYER:-"6"}
EPOCHS=${EPOCHS:-3}
OUT=${OUT:-"results/phase_b"}
mkdir -p "$OUT" logs
GPU_ARR=($GPUS)
N_GPU=${#GPU_ARR[@]}
i=0
for SEED in $SEEDS; do
  GPU=${GPU_ARR[$((i % N_GPU))]}
  LOG="logs/phase_b_seed${SEED}_gpu${GPU}.log"
  echo "[launch] Phase B seed=$SEED gpu=$GPU → $LOG"
  CUDA_VISIBLE_DEVICES=$GPU python src/phase_b_training_residual.py \
    --seed "$SEED" --device "cuda:0" --model "$MODEL" --layer "$LAYER" \
    --epochs "$EPOCHS" --out-dir "$OUT" > "$LOG" 2>&1 &
  i=$((i + 1))
  if (( i % N_GPU == 0 )); then wait; fi
done
wait
echo "[done] Phase B multi-seed complete."
