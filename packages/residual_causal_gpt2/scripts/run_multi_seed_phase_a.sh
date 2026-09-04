#!/usr/bin/env bash
# Multi-seed Phase A (frozen residual ports) across parallel GPUs
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
GPUS=${GPUS:-"0"}
MODEL=${MODEL:-"gpt2"}
LAYER=${LAYER:-"6"}
TASK=${TASK:-"modular"}
OUT=${OUT:-"results/phase_a"}
N_SIG=${N_SIG:-96}
N_HELD=${N_HELD:-96}
mkdir -p "$OUT" logs
GPU_ARR=($GPUS)
N_GPU=${#GPU_ARR[@]}
i=0
for SEED in $SEEDS; do
  GPU=${GPU_ARR[$((i % N_GPU))]}
  LOG="logs/phase_a_seed${SEED}_gpu${GPU}.log"
  echo "[launch] seed=$SEED gpu=$GPU → $LOG"
  CUDA_VISIBLE_DEVICES=$GPU python src/phase_a_frozen_ports.py \
    --seed "$SEED" --device "cuda:0" --model "$MODEL" --layer "$LAYER" \
    --task "$TASK" --n-sig "$N_SIG" --n-held "$N_HELD" --out-dir "$OUT" \
    > "$LOG" 2>&1 &
  i=$((i + 1))
  if (( i % N_GPU == 0 )); then wait; fi
done
wait
echo "[done] Phase A multi-seed complete. Results in $OUT/"
python scripts/aggregate_seeds.py --phase a --in-dir "$OUT" || true
