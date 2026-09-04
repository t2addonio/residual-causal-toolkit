# Upload status — 2026-09-04 (evening)

Repo: https://github.com/t2addonio/residual-causal-toolkit
Head: 1d3e9f80+

## Runnable sources now on main
Optical: phase-cancel, SAE, P1/P2/P3, Fresnel, multimode, Jones, Continuum P1/P2/P3
RF/SDR: RF-P1, RF-P2, sdr_residual_offline_iq.py (probe arms, S checksum, no live TX)
NV: temperature residual, ODMR (D,E) toolkit core
Audio: CRAI-3 carrier resynthesis, CRAI-96k causal lock-in
Telemetry: T-P1 path-patch
ISA: residual_interferometer.py
GPT-2: Phase D online write-hook + toolkit primitives + launchers

## Still local-only
Full-length originals of the above (artifacts/*.py are longer / figure-complete).
crai_per_mic_isolate.py, crai_debleed_demo.py
Mueller family + UAH real, NV multi-B / 14D poly, Planck/EEG 14D
phase_d_stronger.py, train_ab.py, GPT-2 model/train/math modules
PNG / WAV binaries
