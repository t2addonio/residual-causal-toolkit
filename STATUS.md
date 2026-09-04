# Upload status — 2026-09-04

Repo: https://github.com/t2addonio/residual-causal-toolkit

## On GitHub (runnable sources)
- experiments/isa/residual_interferometer.py — PHASE(φ)+BEAM(θ) unitary 2-D core
- experiments/optical/light_residual_phase_cancel.py
- experiments/optical/light_residual_sae.py
- experiments/optical/light_residual_path_patch_p1.py
- experiments/optical/light_residual_fno_continuum_p1.py — field port + kernel port ΔR
- experiments/rf/rf_residual_path_patch_p1.py
- experiments/rf/rf_residual_path_patch_p2.py
- experiments/nv/nv_residual_temperature.py
- experiments/audio/crai_96k_causal.py — 0.333 ms @ 32-sample / 96 kHz
- experiments/telemetry/telemetry_residual_path_patch.py
- packages/residual_causal_gpt2/phase_d_online_residual.py — 16/16 write-hook
- packages/residual_causal_gpt2/residual_toolkit/{hooks,causal_filters,pure_quad,interferometer}.py
- packages/residual_isa_phase_e/ launchers + aggregator
- packages/residual_train_ab_filter/ launchers (Train A/B honest FAIL already documented)

## Still in working store only (next batches)
Full-length originals remain under /home/workdir/artifacts:
- crai_clean_isolate.py, crai_per_mic_isolate.py, crai_debleed_demo.py
- light_residual_fno_continuum_p2.py / p3.py
- light_residual_{fresnel,multimode,polarization,mueller*}.py
- nv_residual_odmr_toolkit.py, nv_residual_deep_multi_B.py
- sdr_residual_offline_iq.py
- residual_causal_gpt2/phase_d_stronger.py, train_ab.py, model/train/math modules
- PNG / WAV binaries (not pushed; listed in experiment summaries)

GitHub MCP push is text-per-commit. Binaries stay local. Full-length scripts
that exceed a comfortable commit payload are landing as protocol-complete
runnable cores first; byte-identical originals follow as the queue allows.
