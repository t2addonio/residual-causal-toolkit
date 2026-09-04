# Upload status — 2026-09-04 late afternoon

https://github.com/t2addonio/residual-causal-toolkit

## Now on main (runnable cores + several full modules)
Optical: phase-cancel, SAE, P1–P3, Fresnel, multimode, Jones, Continuum P1–P3,
         Mueller Stokes, Mueller ΔM diattenuator, Mueller retarder
RF/SDR: RF-P1, RF-P2, sdr_residual_offline_iq.py
NV: temperature, ODMR (D,E), multi-NV + B-field
Audio: CRAI-3, CRAI-96k
Telemetry: T-P1
ISA: residual_interferometer.py
GPT-2: MiniGPT model.py + train.py + residual_math.py (R1/R2/R3),
        Phase D online write-hook, Phase D-stronger mapper, Train A/B result lock

## Still local
crai_per_mic_isolate.py, crai_debleed_demo.py (full-length)
UAH real Mueller, Planck/EEG 14D scripts, audio_residual_demo.py
PNG / WAV binaries
Full-length GPU trainers (phase_d_stronger 21k, train_ab 14k) — protocol cores are on GitHub;
byte-identical originals remain in the working store.
