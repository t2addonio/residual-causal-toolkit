#!/usr/bin/env bash
# Phase E — 16-seed online residual interferometer on 4x RTX 4090
set -euo pipefail
cd "$(dirname "$0")"
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"}
GPUS=${GPUS:-"0 1 2 3"}
LAYER=${LAYER:-"6"}
OUT=${OUT:-"results/phase_e"}
N_TRAIN=${N_TRAIN:-"8192"}
EPOCHS=${EPOCHS:-"12"}
BATCH=${BATCH:-"16"}
N_SIG=${N_SIG:-"64"}
N_HELD=${N_HELD:-"64"}
N_PHI=${N_PHI:-"9"}
mkdir -p "$OUT" logs
GPU_ARR=($GPUS)
N_GPU=${#GPU_ARR[@]}
i=0
echo "[Phase E] seeds=[$SEEDS] gpus=[$GPUS] n_train=$N_TRAIN epochs=$EPOCHS n_phi=$N_PHI"
for SEED in $SEEDS; do
  GPU=${GPU_ARR[$((i % N_GPU))]}
  LOG="logs/phase_e_seed${SEED}_gpu${GPU}.log"
  echo "[launch] Phase E seed=$SEED gpu=$GPU → $LOG"
  CUDA_VISIBLE_DEVICES=$GPU python -u phase_e_interferometer.py \
    --seed "$SEED" --device "cuda:0" --layer "$LAYER" \
    --n-train "$N_TRAIN" --epochs "$EPOCHS" --batch-size "$BATCH" \
    --n-sig "$N_SIG" --n-held "$N_HELD" --n-phi "$N_PHI" --out-dir "$OUT" \
    > "$LOG" 2>&1 &
  i=$((i + 1))
  if (( i % N_GPU == 0 )); then wait; fi
done
wait
echo "[done] Phase E 4-GPU multi-seed complete. Results in $OUT/"
python aggregate_phase_e.py --root "$OUT" --out "$OUT/phase_e_aggregate.json"
