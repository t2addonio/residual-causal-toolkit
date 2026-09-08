# Causal Phase and Magnitude Interventions on Quantum Residual Streams

**Tony Taddonio**  
Independent researcher · t2addonio@gmail.com  
August 2026

## Abstract

Residual-stream phase and magnitude interventions isolate load-bearing structure in classical transformers, including a causally necessary residual component after grokking [Taddonio, 2026a]. This note asks whether the same intervention logic transfers when the residual object is the mid-circuit state of a hybrid variational circuit.

A 4-qubit hybrid QNN is trained, then a frozen RZ(θ) or RY(θ) is swept. Phase ablation produces a robust non-monotonic causal response across 32 seeds (mean max ΔAcc ≈ 0.23; 56% of seeds exceed a 0.20 drop). Pauli residual difference magnitude tracks performance at r ≈ 0.97 and is low-rank (top-3 PCs ≈ 75%). Every qubit is load-bearing under both axes; phase sensitivity increases toward later qubits; magnitude is strong and comparatively uniform.

Full experimental write-up (methods, Stage 1–4, tables, figure guide, limitations) is in the working-store file `Quantum_Residual_Stream_Causal_Interventions_v1.md` and is folded into §10 of the unified paper.

## Key numbers

- Stage 2 mean max ΔAcc: 0.228 ± 0.109; median 0.219; >0.20 in 56.2%; >0.25 in 37.5%.
- Stage 3 max mean residual shift 0.129 at θ ≈ 3.01; PC1 variance 0.397; top-3 cumulative 0.749; corr(‖Δr‖, acc drop) 0.972.
- Stage 4 phase ΔAcc by qubit: [0.204, 0.254, 0.267, 0.333]. Magnitude: [0.318, 0.302, 0.293, 0.328].

Caveat: 4 qubits, statevector, synthetic labels. Method transfer, not large-model phenomenology. 14-D quantum residual in the toolkit used reconstructed Pauli geometry, not a live QNN dump.
