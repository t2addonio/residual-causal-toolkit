# Residual Causal Toolkit

Cross-domain residual-stream addressing: contrastive extraction, path-patch, parallel ports, Residual Stream ISA, CRAI audio isolation, and field / kernel ports.

**Author:** Tony Taddonio
**Repositories:**
- This repo — full toolkit corpus (created 4 September 2026)
- [`t2addonio/residual-stream-grokking`](https://github.com/t2addonio/residual-stream-grokking) — original grokking residual-component paper (left intact)

**Progress paper (4 Sep 2026):** [`docs/papers/Residual_Causal_Toolkit_Progress_Paper_2026-09.md`](docs/papers/Residual_Causal_Toolkit_Progress_Paper_2026-09.md)

## One-line claim

A residual component in an additive mixture is recoverable by contrastive extraction, causally necessary under path-patch / zero / random / atom, and addressable on a bus that is never overwritten. Processors (transformer block, Fresnel, Mueller, frozen Fourier multiplier) sit on that bus. They are not a new ISA.

## What this is / is not

It is an addressing filter plus a small instruction set on residual RAM:

`LOAD  EXTRACT  FILTER  STORE  PHASE  BEAM  READ  FAULT`

It is not a neural-operator lab, not an LLM, not a QPU, and not a replacement for physics you can already write (Fresnel, lock-in, Mueller, CRAI DSP).

## Protocol (5 steps + 1b)

1. Define the mixture — carrier + residual; signal vs held ensembles.
1b. If and only if the residual is a field — keep domain Ω vs fiber F separate; name processor G; open kernel port ΔR only when it exists. Skip 1b on vector buses.
2. Contrastive extraction — mean difference, probe arms, SVD (diagnostic), 14-D pure-quad, residual-seeded dictionary / SAE.
3. Causal filter — path-patch, zero, random, atom. Keep only ports that pass. Complement and original `S` untouched.
4. Selective control — STORE / PHASE / BEAM on a copy, or an online write-hook.
5. Validate — residual-dependent READ moves; random intact; checksum of `S` unchanged except at the controlled write site.

## Repo layout

```
docs/papers/          progress paper + protocol notes
experiments/          domain scripts + run summaries
  optical/            phase-cancel, Fresnel, Jones, Mueller, continuum P1–P3
  rf/                 RF-P1/P2/P3, 5-D, 14-D, offline SDR IQ
  nv/                 ODMR toolkit, B, T, 14-D, parallel ports
  cmb/                synthetic CMB ports + real Planck
  eeg/                question residual, 5-D, 14-D
  audio/              residual demos + CRAI-1/2/3 + 96 kHz causal bank
  telemetry/          synthetic boat-style + real UCI naval
  transformer/        modular / tiny-transformer 14-D
  quantum/            reconstructed QNN residual 14-D
  isa/                interferometer + Residual Stream ISA
  logical/            logical residual stream v0 + NV
  core/               parallel ports, 14-D campaign, torsion, triples
packages/
  residual_causal_gpt2/     Phases A–D + toolkit library
  residual_isa_phase_e/     Phase E PHASE(φ) package
  residual_train_ab_filter/ Train A/B (B was not ≥ A)
  sdr_capture/              RTL-SDR V4 offline IQ examples
INDEX.md                    file-by-file catalog
```

Figures (`.png`) and demonstration audio (`.wav`) are listed in `INDEX.md`. They live in the working artifacts store and can be regenerated from the scripts. The GitHub connector used for this upload is text-first; binaries were not force-pushed.

## Status on 4 September 2026

**Closed.** Optical classical suite including real UAH Mueller. RF path-patch P1–P3. NV single / multi / B / T plus real Hole/NoHole 14-D. Planck residual ports. EEG. Telemetry synthetic + real naval. GPT-2 Phases A–D (16/16 online write-hook). Residual Stream ISA. CRAI-3 synthetic PASS + 96 kHz causal bank. Continuum P1–P3. Train A/B executed; success rule FAIL (`B` is not `≥ A`).

**In hand.** RTL-SDR Blog V4 (R828D + RTL2832U), receive-only, arrived 3 September 2026. Preferred first plant: conducted coax + pads, SDR as RX, separate scriptable TX. Closed loop = READ residual port → FILTER → WRITE an actuator. Received `S` untouched.

**Audio product gate.** Trained-ear isolation bar, not computer metrics alone.

## How to run a slice

```bash
# Optical path-patch P1
python experiments/optical/light_residual_path_patch_p1.py

# CRAI-3 clean isolation (synthetic scene)
python experiments/audio/crai_clean_isolate.py

# Offline SDR IQ residual
python experiments/rf/sdr_residual_offline_iq.py

# GPT-2 Phase D online write-hook
cd packages/residual_causal_gpt2
python phase_d_online_residual.py
```

Train A/B (already run; do not change architecture between arms):

```bash
cd packages/residual_train_ab_filter
cat COMMANDS.txt
```

## Related paper

Taddonio, *A Causally Necessary Residual-Stream Component in Grokking, Distinct from Known Algorithmic Features* — [`residual-stream-grokking`](https://github.com/t2addonio/residual-stream-grokking).

## License

MIT. See `LICENSE`.
