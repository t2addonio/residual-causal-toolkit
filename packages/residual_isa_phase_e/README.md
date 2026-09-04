# Residual ISA — Phase E (online interferometer)

Portable residual address space + one rotate instruction, run on GPT-2.

| File | Role |
|---|---|
| `phase_e_interferometer.py` | Train modular GPT-2, extract 2-D core, sweep PHASE(φ) online |
| `run_phase_e_4x4090.sh` | 16-seed launcher, 4 GPUs, then aggregate |
| `aggregate_phase_e.py` | Seed summary JSON |
| `RUN.txt` | Exact commands for the 4090 box |
| `SDR_NEXT_WEEK.txt` | Analog drop-in when the SDR arrives |
| `requirements.txt` | torch, transformers, numpy |

This is not a QPU and not a new chip. The residual stream is RAM.
Surviving ports are addresses. PHASE(φ) is the only new instruction.

See `RUN.txt`.
