# Residual Causal Toolkit

Cross-domain residual-stream addressing: contrastive extraction, path-patch, parallel ports, Residual Stream ISA, CRAI audio isolation, and field / kernel ports.

**Author:** Tony Taddonio  
**Repositories:**
- This repo — full toolkit corpus (created 4 September 2026)
- [`t2addonio/residual-stream-grokking`](https://github.com/t2addonio/residual-stream-grokking) — original grokking residual-component paper (left intact)

**Unified paper (8 Sep 2026):** [`docs/papers/Residual_Causal_Toolkit_Unified_Paper_2026-09.md`](docs/papers/Residual_Causal_Toolkit_Unified_Paper_2026-09.md)

Illustrated 25-page PDF with 29 campaign figures lives in the working artifacts store as `Residual_Causal_Toolkit_Unified_Paper_2026-09.pdf` (GitHub connector is text-first; binaries stay in the working store).

**Quantum companion:** [`docs/papers/Quantum_Residual_Stream_Causal_Interventions_v1.md`](docs/papers/Quantum_Residual_Stream_Causal_Interventions_v1.md)

**Earlier progress paper (4 Sep 2026):** [`docs/papers/Residual_Causal_Toolkit_Progress_Paper_2026-09.md`](docs/papers/Residual_Causal_Toolkit_Progress_Paper_2026-09.md)

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

## Status on 8 September 2026

**Closed.** Optical classical suite including real UAH Mueller. RF path-patch P1–P3. NV single / multi / B / T plus real Hole/NoHole 14-D. Planck residual ports. EEG. Telemetry synthetic + real naval. GPT-2 Phases A–D (16/16 online write-hook). Residual Stream ISA. CRAI-3 synthetic PASS + 96 kHz causal bank. Continuum P1–P3. Train A/B executed; success rule FAIL (`B` is not `≥ A`). Unified paper written (optical + superposition + quantum + ISA + later campaign).

**In hand.** RTL-SDR Blog V4 (R828D + RTL2832U), receive-only, arrived 3 September 2026. First IQ dump 7 September 2026 (`fm_sanity.cu8` at 97.5 MHz). Preferred first plant: conducted coax + pads, SDR as RX, separate scriptable TX. Closed loop = READ residual port → FILTER → WRITE an actuator. Received `S` untouched.

**Audio product gate.** Trained-ear isolation bar, not computer metrics alone.

## Related paper

Taddonio, *A Causally Necessary Residual-Stream Component in Grokking, Distinct from Known Algorithmic Features* — [`residual-stream-grokking`](https://github.com/t2addonio/residual-stream-grokking).

## License

MIT. See `LICENSE`.
