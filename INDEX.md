# Residual Causal Toolkit — repository index

Public repo: https://github.com/t2addonio/residual-causal-toolkit

S is never overwritten. Interventions live on parallel ports.
FILTER = path-patch + zero + random + (atom). Probe arms, not co-occurrence SVD-1.

## Unified paper (8 September 2026)

- `docs/papers/Residual_Causal_Toolkit_Unified_Paper_2026-09.md` — text source
- Working store: `Residual_Causal_Toolkit_Unified_Paper_2026-09.pdf` — 25 pages, 29 campaign figures
- `docs/papers/Quantum_Residual_Stream_Causal_Interventions_v1.md` — quantum companion folded into §10

## Companion repos

- Paper / grokking discovery: https://github.com/t2addonio/residual-stream-grokking
- Isolate + rescue + freeze-at-transition: https://github.com/t2addonio/isolate-rescue-grokking
- Hybrid lab: https://github.com/t2addonio/hybrid-lab

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

PNG figures, WAV renders, measured IQ captures, UAH 4×4×Nλ cubes, GPT-2 checkpoints, the illustrated 25-page PDF (~5.5 MB).
Byte-identical figure-complete originals remain in the working store.
