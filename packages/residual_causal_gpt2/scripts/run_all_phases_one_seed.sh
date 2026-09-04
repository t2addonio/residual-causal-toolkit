#!/usr/bin/env bash
# Sequential Phase A → B → C for a single seed
set -euo pipefail
SEED="${1:-0}"
DEVICE="${2:-cuda}"
LAYER="${LAYER:-6}"
echo "Running all phases for seed=$SEED device=$DEVICE layer=$LAYER"
python run_phase_a_frozen_ports.py --seed "$SEED" --device "$DEVICE" --layer "$LAYER" --finetune_steps 200 --out_dir runs/phase_a
python run_phase_b_training_residual.py --seed "$SEED" --device "$DEVICE" --layer "$LAYER" --steps 300 --ckpt_every 50 --out_dir runs/phase_b
python run_phase_c_closed_loop.py --seed "$SEED" --device "$DEVICE" --layer "$LAYER" --finetune_steps 200 --out_dir runs/phase_c
echo "Done seed=$SEED"
