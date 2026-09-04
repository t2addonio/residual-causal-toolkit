# Residual Causal Toolkit — index

Generated 2026-09-04 for `t2addonio/residual-causal-toolkit`.

The original grokking paper repo `t2addonio/residual-stream-grokking` is left intact.

## Papers

- Working artifacts store: `Residual_Causal_Toolkit_Progress_Paper_2026-09.docx` (Word) and `.md` (markdown)
- Also in artifacts: methodology, ISA one-pager, Phase D lock-in, optical paper, superposition synthesis, handoff summaries

## Domain map (working artifacts → intended repo path)

| Domain | Prefix | What closed |
|---|---|---|
| Optical | `light_residual_*` | phase-cancel, Fresnel, multimode, Jones, SAE, path-patch P1–P3, Mueller family, real UAH, retarder, Continuum P1–P3 |
| RF | `rf_residual_*`, `sdr_*` | RF-P1/P2/P3, 5-D, 14-D, offline SDR IQ |
| NV | `nv_residual_*` | single ODMR, multi-NV, B, T, real Hole/NoHole, 14-D, parallel ports |
| CMB | `cmb_*`, `planck_*` | synthetic ports, real Planck vs ΛCDM, seeded SAE, 14-D |
| EEG | `eeg_residual_*` | question residual, 5-D change, 14-D |
| Audio | `audio_*`, `crai_*` | residual demo, CRAI-1/2/3, 96 kHz causal bank |
| Telemetry | `telemetry_*` | synthetic boat-style, real UCI naval |
| Transformer | `transformer_*`, `modular_*`, `packages/residual_causal_gpt2/` | 14-D, GPT-2 Phases A–D |
| Quantum | `quantum_*`, `qnn_*` | reconstructed QNN 14-D |
| ISA | `residual_interferometer*`, `packages/residual_isa_phase_e/` | PHASE/BEAM core, Phase E package |
| Logical | `residual_logical_stream_*` | v0 + real NV consensus layer |
| Core | `residual_toolkit_*`, `residual_poly_*`, `residual_14d_*` | parallel ports, 14-D campaign |
| Train A/B | `packages/residual_train_ab_filter/` | executed; B was not ≥ A |
| SDR plant | `packages/sdr_capture/` | V4 arrived 3 Sep 2026; offline IQ first |

## Counts in the working store

280 files after excluding cache/preview: 73 py, 82 txt, 58 png, 27 wav, 14 sh, 9 docx, plus npy/cf32/gz.

GitHub upload path used here is text-first. Scripts and summaries are staged for this repo; png/wav/npy/docx stay in the working artifacts store and are regenerable from the scripts.

## How to resume from this repo

```bash
# package slices already in tree
cat packages/residual_train_ab_filter/COMMANDS.txt
cat packages/residual_isa_phase_e/RUN.txt
cat packages/sdr_capture/CAPTURE.txt
```
