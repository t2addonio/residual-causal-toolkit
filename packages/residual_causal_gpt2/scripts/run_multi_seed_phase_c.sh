#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
GPUS=${GPUS:-"0"}
MODEL=${MODEL:-"gpt2"}
LAYER=${LAYER:-"6"}
PORT=${PORT:-"R2_svd_0"}
OUT=${OUT:-"results/phase_c"}
mkdir -p "$OUT" logs
GPU_ARR=($GPUS)
N_GPU=${#GPU_ARR[@]}
i=0
for SEED in $SEEDS; do
  GPU=${GPU_ARR[$((i % N_GPU))]}
  LOG="logs/phase_c_seed${SEED}_${PORT}_gpu${GPU}.log"
  echo "[launch] Phase C seed=$SEED port=$PORT gpu=$GPU → $LOG"
  CUDA_VISIBLE_DEVICES=$GPU python src/phase_c_closed_loop.py \
    --seed "$SEED" --device "cuda:0" --model "$MODEL" --layer "$LAYER" \
    --port "$PORT" --out-dir "$OUT" > "$LOG" 2>&1 &
  i=$((i + 1))
  if (( i % N_GPU == 0 )); then wait; fi
done
wait
echo "[done] Phase C multi-seed complete (port=$PORT)."
python scripts/aggregate_seeds.py --phase c --in-dir "$OUT" --port "$PORT" || true
