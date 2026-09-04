# Residual Causal Toolkit — GPT-2

Cross-domain residual causal toolkit applied to GPT-2 residual streams.

## Phases

| Phase | What | Script |
|-------|------|--------|
| **A** | Frozen residual ports + causal filter (R1/R2/R3) | `src/phase_a_frozen_ports.py` |
| **B** | Residual trajectory during training | `src/phase_b_training_residual.py` |
| **C** | Offline residual closed-loop (path-patch residual coefficient) | `src/phase_c_closed_loop.py` |
| **D** | **Online** residual control (write hook during forward) | `src/phase_d_online_residual_control.py` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers tqdm numpy
```

## 4-GPU multi-seed execution

```bash
# Phase A — residual ports
SEEDS="0 1 2 3 4 5 6 7" GPUS="0 1 2 3" bash scripts/run_multi_seed_phase_a.sh

# Phase C — offline residual closed-loop (high residual-power port)
PORT=R2_svd_0 SEEDS="0 1 2 3 4 5 6 7" GPUS="0 1 2 3" bash scripts/run_multi_seed_phase_c.sh

# Phase D — online residual control
SEEDS="0 1 2 3 4 5 6 7" GPUS="0 1 2 3" bash scripts/run_multi_seed_phase_d.sh
```

## Single-seed smoke tests

```bash
python src/phase_a_frozen_ports.py --seed 0 --device cuda:0 --layer 6
python src/phase_c_closed_loop.py --seed 0 --device cuda:0 --layer 6 --port R2_svd_0
python src/phase_d_online_residual_control.py --seed 0 --device cuda:0 --layer 6
```

## Standalone Phase D (no package imports)

```bash
python phase_d_online_residual.py --seed 0 --device cuda:0 --layer 6
```

## Aggregate

```bash
python scripts/aggregate_seeds.py --phase a --in-dir results/phase_a
python scripts/aggregate_seeds.py --phase c --in-dir results/phase_c --port R2_svd_0
python scripts/aggregate_phase_d.py --root results/phase_d
```
