#!/usr/bin/env bash
# Paired Arm A / Arm B multi-seed on 4 GPUs.
set -euo pipefail
cd "$(dirname "$0")"
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7"}
GPUS=${GPUS:-"0 1 2 3"}
ARMS=${ARMS:-"A B"}
LAYER=${LAYER:-6}
EPOCHS=${EPOCHS:-6}
N_TRAIN=${N_TRAIN:-4096}
BATCH=${BATCH:-16}
OUT=${OUT:-results}
MODEL=${MODEL:-gpt2}
mkdir -p "$OUT" logs
GPU_ARR=($GPUS)
N_GPU=${#GPU_ARR[@]}
echo "[ab] seeds=[$SEEDS] arms=[$ARMS] gpus=[$GPUS] layer=$LAYER epochs=$EPOCHS"
i=0
for SEED in $SEEDS; do
  for ARM in $ARMS; do
    GPU=${GPU_ARR[$((i % N_GPU))]}
    LOG="logs/arm${ARM}_seed${SEED}_gpu${GPU}.log"
    echo "[launch] arm=$ARM seed=$SEED gpu=$GPU → $LOG"
    CUDA_VISIBLE_DEVICES=$GPU python -u train_ab.py \
      --arm "$ARM" --seed "$SEED" --device cuda:0 --layer "$LAYER" \
      --epochs "$EPOCHS" --n-train "$N_TRAIN" --batch-size "$BATCH" \
      --model "$MODEL" --out-dir "$OUT" > "$LOG" 2>&1 &
    i=$((i + 1))
    if (( i % N_GPU == 0 )); then wait; fi
  done
done
wait
python aggregate.py --results "$OUT" --out "$OUT/ab_summary.json" --md "$OUT/ab_summary.md"
echo "[ab] done. see $OUT/ab_summary.md"
