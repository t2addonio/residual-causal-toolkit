# Residual Causal Toolkit — repository index

Public repo: https://github.com/t2addonio/residual-causal-toolkit
Working store (figures, WAVs, full-length originals): `/home/workdir/artifacts`

S is never overwritten. Interventions live on parallel ports.
FILTER = path-patch + zero + random + (atom). Probe arms, not co-occurrence SVD-1.

## Companion repos

- Paper / grokking discovery: https://github.com/t2addonio/residual-stream-grokking
- Isolate + rescue + freeze-at-transition (16-seed modular add): https://github.com/t2addonio/isolate-rescue-grokking
- Hybrid lab: https://github.com/t2addonio/hybrid-lab

## Layout

```
experiments/
  optical/     phase-cancel, SAE, P1–P3, Fresnel, multimode, Jones,
               Mueller Stokes / ΔM / retarder / UAH real, Continuum P1–P3
  rf/          RF-P1, RF-P2, SDR offline IQ ISA, 14D RF summary
  nv/          ODMR (D,E), temperature, multi-NV + B, 14D E²
  cmb/         Planck residual ports + 14D + seeded-SAE notes
  eeg/         question demo + 5D + 14D summaries
  audio/       CRAI-1 debleed, CRAI-2 per-mic, CRAI-3 resynthesis, CRAI-96k, 1250 Hz demo
  telemetry/   T-P1 NMEA + vibration 3.3× order
  isa/         residual interferometer PHASE(φ)+BEAM(θ)
  logical/     R1/R2/R3 consensus layer (v0 + real NV)
  core/        shared 14D map + parallel-ports / torsion notes
  quantum/     14D QNN / mid-circuit residual notes
packages/
  residual_causal_gpt2/   MiniGPT + Phases A–D + D-stronger + launchers
  residual_train_ab_filter/  Arm A vs B (honest FAIL 2026-08-31)
  residual_isa_phase_e/   Phase E interferometer launcher
  sdr_capture/            RTL-SDR V4 offline IQ recipe
docs/papers/              progress paper markdown
```

## Protocol spine (what to run first)

1. `experiments/optical/light_residual_phase_cancel.py` — α=1 nulls, α=2 sign-flip
2. `experiments/optical/light_residual_path_patch_p1.py` — mediation, not correlation
3. `experiments/rf/rf_residual_path_patch_p1.py` + `sdr_residual_offline_iq.py`
4. `experiments/nv/nv_residual_odmr_toolkit.py` then `nv_residual_poly_14d_real_summary.txt`
5. `experiments/audio/crai_clean_isolate.py` (printable) then `crai_96k_causal.py` (live)
6. `packages/residual_causal_gpt2/phase_d_online_residual.py` — 16/16 write-hook
7. `experiments/isa/residual_interferometer.py` — unitary plane, complement flat

## Locks (do not quietly reverse)

- Path-patch is the causality test. Online write-hook is the non-redundant next claim.
- Neural operator is a processor on the Residual Stream ISA, not a replacement ISA.
- Continuum P2: single-feature probe arms when residuals co-occur. Not SVD-1.
- Continuum P3: orthogonalize source arms against the carrier before hard-zero.
- CRAI product: toolkit DSP only — no net, no weights, no GPU inference. No sidechain mic.
- 96 kHz: STFT insert latency is illegal. Causal lock-in, 0.333 ms @ 32-sample block.
- Audio product gate is Tony's trained ear, not computer metrics alone.
- SDR: receive-only V4. Conducted first plant. No live TX rewrite until offline FILTER passes on files the radio wrote.
- Train A/B 2026-08-31: B is not ≥ A. Do not invent a third trainer architecture.

## Not in git (on purpose)

PNG figures, WAV renders, measured IQ captures, UAH 4×4×Nλ cubes, GPT-2 checkpoints.
Byte-identical 15–26 k figure-complete originals remain in the working store.
