# The Residual Causal Toolkit

**Superposition as Mixed-Signal Geometry, a Residual Stream ISA, and Substrate-Agnostic Control Across Physical and Computational Buses**

Tony Taddonio  
Independent researcher · t2addonio@gmail.com  
8 September 2026 · Working paper

Companion repositories: [`t2addonio/residual-causal-toolkit`](https://github.com/t2addonio/residual-causal-toolkit) and [`t2addonio/residual-stream-grokking`](https://github.com/t2addonio/residual-stream-grokking).

Illustrated PDF with campaign figures: `Residual_Causal_Toolkit_Unified_Paper_2026-09.pdf` in the working artifacts store (25 pages, 29 figures).

---

## Abstract

We present a residual causal toolkit that treats residual streams as additive mixed-signal objects in which residual components exist in superposition with dominant carriers. The toolkit operationalizes a domain-agnostic protocol — mixture definition, contrastive extraction, a four-part causal filter (path-patch, zero, random, atom), and selective control — together with a Residual Stream ISA that names legal addresses on a bus that is never overwritten. Processors (transformer blocks, Fresnel propagators, Mueller matrices, frozen Fourier multipliers, lock-in banks) sit on that bus. They are not a new instruction set.

The same structural pattern transfers from discrete transformer residual streams through narrowband RF mixtures, pure classical optical residual fields (synthetic and real Mueller), nitrogen-vacancy color-center ODMR, Planck CMB residuals versus ΛCDM, multi-channel telemetry, EEG, audio STFT / CRAI isolation, an FNO-form continuum processor, and mid-circuit Pauli residuals of a hybrid quantum circuit. Across those domains residual structure is low-rank, often dominated by a pure second-moment component (energy² / E² / residual-power²), and is selectively nullable while the carrier and the original stream remain intact.

Two honesty constraints are locked. Computer PASS is supporting evidence; on audio the product gate is a trained ear. A neural operator is a processor on the Residual Stream ISA, not a replacement ISA. A same-architecture Train A/B test on modular GPT-2 did not show that putting a residual READ in the training loop beats cross-entropy alone. Hardware now in hand includes an RTL-SDR Blog V4 (receive-only); the first owned-plant closed loop is defined and not yet run.

**Keywords.** residual streams · superposition · path-patch · mechanistic interpretability · optical Mueller calculus · NV-center ODMR · RF mixed-signal · Residual Stream ISA · CRAI

---

## 1. Introduction

In mechanistic interpretability the residual stream of a transformer is the central object in which features live in superposition [Elhage et al., 2021; Nanda et al., 2023]. Prior work isolated a causally necessary residual-stream component that persists after grokking and is nearly orthogonal to known algorithmic features [Taddonio, 2026a]. The present corpus asks a larger question: is that geometry an idiosyncrasy of neural activations, or is it the generic structure of an additive residual stream?

A residual stream can be written

```
x  =  carrier  +  residual  (+ noise)
```

and the residual component remains causally necessary for residual-dependent observables even after the carrier is accounted for. This paper unifies four earlier working documents — the optical residual-streams paper, the superposition synthesis, the quantum residual-intervention note, and the Residual Stream ISA one-pager — with the campaign through 8 September 2026.

The contribution is methodological and experimental. We do not claim that every residual-like signal in the wild is path-patchable. We claim that when a residual component is present in an additive mixture and the ensembles are properly defined, the same addressing filter recovers and controls it, and that the original stream can be left untouched while parallel ports do the work.

See the full illustrated PDF in the working store for figures 1–29 and the complete campaign tables (optical P1–P3, RF-P1/P2, NV E / multi / B / T, telemetry, EEG, Planck, GPT-2 Phase D, CRAI-3, Continuum P1–P3, Train A/B, SDR V4).

## Protocol spine

1. Define the mixture. 1b if the residual is a field.
2. Contrastive extraction (probe arms, not co-occurrence SVD-1).
3. Causal filter: path / zero / random / atom.
4. STORE / PHASE / BEAM or online write-hook.
5. Validate. S checksum unchanged except at the controlled write.

ISA: LOAD EXTRACT FILTER STORE PHASE BEAM READ FAULT.

## Locks

Path-patch is the causality test. S is RAM. Audio product gate is a trained ear. Neural operators process the bus; they do not replace FILTER. Train A/B 2026-08-31: B is not ≥ A.

## Status 8 September 2026

Closed: optical including real Mueller; RF P1–P3; NV including real 14-D E²; Planck; EEG; telemetry; GPT-2 A–D 16/16; ISA; CRAI-3 synthetic PASS + 96 kHz lock-in; Continuum P1–P3; Train A/B honest FAIL; first owned-SDR IQ dump.

Next: owned-plant RF closed loop on the V4; real close-mics through CRAI before any plugin wrap.

Full text of the 25-page working paper, with all tables and figure captions, is in the working-store PDF and in the long-form markdown kept next to it in artifacts.
